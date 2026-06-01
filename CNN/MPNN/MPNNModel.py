import torch
import torch.nn as nn

from MPNNData import ECN_WEIGHT_SRC_IDX


class EdgeNet(nn.Module):
    """Two-layer MLP that produces a learned representation for each edge.

    Input:  concatenation of [center_atom_fea, neighbor_atom_fea, bond_fea]
    Output: edge representation of dimension atom_fea_len (same dim as atom features
            so that residual addition in MPNNConvLayer is straightforward).
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, edge_hidden_dim: int):
        super().__init__()
        in_dim = 2 * atom_fea_len + nbr_fea_len
        self.fc1 = nn.Linear(in_dim, edge_hidden_dim)
        self.norm1 = nn.LayerNorm(edge_hidden_dim)
        self.fc2 = nn.Linear(edge_hidden_dim, atom_fea_len)
        self.norm2 = nn.LayerNorm(atom_fea_len)
        self.act = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm1(self.fc1(x)))
        x = self.act(self.norm2(self.fc2(x)))
        return x


class MPNNConvLayer(nn.Module):
    """Single message-passing layer with a learned EdgeNet and configurable aggregation.

    Variant A (aggregation='ecn_weighted'):
        Aggregates edge representations weighted by softmax-normalised ECoN weights.
    Variant B (aggregation='attention'):
        Learns scalar attention scores over edge representations, then softmax-aggregates.

    Parameters
    ----------
    atom_fea_len : int
    nbr_fea_len : int
    edge_hidden_dim : int   hidden dimension inside EdgeNet
    aggregation : str       'ecn_weighted' | 'attention'
    ecn_weight_idx : int    column index of ecn_weight_src in the edge feature vector
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, edge_hidden_dim: int,
                 aggregation: str = 'ecn_weighted',
                 ecn_weight_idx: int = ECN_WEIGHT_SRC_IDX):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.aggregation = aggregation
        self.ecn_weight_idx = ecn_weight_idx

        self.edge_net = EdgeNet(atom_fea_len, nbr_fea_len, edge_hidden_dim)
        self.norm_out = nn.LayerNorm(atom_fea_len)
        self.act = nn.Softplus()

        if aggregation == 'attention':
            self.attn_fc = nn.Linear(atom_fea_len, 1)

    def forward(self, atom_in_fea: torch.Tensor,
                nbr_fea: torch.Tensor,
                nbr_fea_idx: torch.LongTensor) -> torch.Tensor:
        """
        Parameters
        ----------
        atom_in_fea  : (N, atom_fea_len)
        nbr_fea      : (N, M, nbr_fea_len)
        nbr_fea_idx  : (N, M)  — local batch atom indices, padded with self-index

        Returns
        -------
        atom_out_fea : (N, atom_fea_len)
        """
        N, M = nbr_fea_idx.shape

        # Gather neighbor atom features -> (N, M, atom_fea_len)
        nbr_atom_fea = atom_in_fea[nbr_fea_idx.view(-1)].view(N, M, self.atom_fea_len)

        # Expand center features -> (N, M, atom_fea_len)
        center_fea = atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len)

        # Edge network input: [center || neighbor || bond] -> (N, M, 2*atom_fea_len + nbr_fea_len)
        edge_input = torch.cat([center_fea, nbr_atom_fea, nbr_fea], dim=2)

        # Learned edge representations -> (N, M, atom_fea_len)
        edge_repr = self.edge_net(edge_input)

        # Padding mask: real edges have at least one non-zero feature value
        pad_mask = (nbr_fea.abs().sum(dim=2) > 0)  # (N, M) bool

        if self.aggregation == 'ecn_weighted':
            # Extract raw ECoN weights -> (N, M)
            ecn_w = nbr_fea[:, :, self.ecn_weight_idx].clamp(min=0.0)
            ecn_w = ecn_w * pad_mask.float()
            # Softmax-normalise over the real neighbourhood
            ecn_w_sum = ecn_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
            norm_w = ecn_w / ecn_w_sum  # (N, M)
            aggregated = (norm_w.unsqueeze(2) * edge_repr).sum(dim=1)  # (N, atom_fea_len)

        elif self.aggregation == 'attention':
            # Learned attention scores -> (N, M, 1)
            scores = self.attn_fc(edge_repr)
            # Mask padding positions to -inf so they don't participate in softmax
            scores = scores.masked_fill(~pad_mask.unsqueeze(2), -1e9)
            attn_w = torch.softmax(scores, dim=1)  # (N, M, 1)
            aggregated = (attn_w * edge_repr).sum(dim=1)  # (N, atom_fea_len)

        else:
            raise ValueError(f"Unknown aggregation: '{self.aggregation}'. "
                             "Choose 'ecn_weighted' or 'attention'.")

        # Residual connection + layer norm + activation
        out = self.act(self.norm_out(aggregated + atom_in_fea))
        return out


class CrystalMPNN(nn.Module):
    """Crystal graph neural network with learned edge representations.

    Replaces the original CGCNN ConvLayer with MPNNConvLayer (EdgeNet + aggregation).
    Architecture otherwise mirrors CrystalGraphConvNet for direct comparison.

    Parameters
    ----------
    orig_atom_fea_len : int   raw node feature dimension from the dataset
    nbr_fea_len       : int   raw edge feature dimension from the dataset
    atom_fea_len      : int   hidden atom feature dimension (post-embedding)
    edge_hidden_dim   : int   hidden dimension inside EdgeNet
    n_conv            : int   number of message-passing layers
    h_fea_len         : int   MLP hidden dimension after global pooling
    n_h               : int   number of MLP layers after pooling
    aggregation       : str   'ecn_weighted' | 'attention'
    """

    def __init__(self, orig_atom_fea_len: int, nbr_fea_len: int,
                 atom_fea_len: int = 64, edge_hidden_dim: int = 128,
                 n_conv: int = 3, h_fea_len: int = 128, n_h: int = 1,
                 aggregation: str = 'ecn_weighted', classification: bool = False):
        super().__init__()

        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        self.convs = nn.ModuleList([
            MPNNConvLayer(atom_fea_len, nbr_fea_len, edge_hidden_dim, aggregation)
            for _ in range(n_conv)
        ])

        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_act = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.fc_acts = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        # 2-class log-softmax head for the SC/non-SC classifier, else scalar regressor.
        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)
        if classification:
            self.dropout = nn.Dropout()
            self.logsoftmax = nn.LogSoftmax(dim=1)
        self.n_h = n_h

    def forward(self, atom_fea: torch.Tensor,
                nbr_fea: torch.Tensor,
                nbr_fea_idx: torch.LongTensor,
                crystal_atom_idx: list) -> torch.Tensor:
        atom_fea = self.embedding(atom_fea)

        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx)

        crys_fea = self._pooling(atom_fea, crystal_atom_idx)

        crys_fea = self.conv_to_fc_act(self.conv_to_fc(crys_fea))

        if self.classification:
            crys_fea = self.dropout(crys_fea)

        if self.n_h > 1:
            for fc, act in zip(self.fcs, self.fc_acts):
                crys_fea = act(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    @staticmethod
    def _pooling(atom_fea: torch.Tensor, crystal_atom_idx: list) -> torch.Tensor:
        """Global mean pooling per crystal."""
        assert sum(len(idx) for idx in crystal_atom_idx) == atom_fea.shape[0]
        return torch.cat([
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ], dim=0)
