import torch
import torch.nn as nn

from data import (ECN_WEIGHT_SRC_IDX, POLY_WEIGHT_IDX,
                      ANGLE_RBF_CENTERS, ANGLE_FEA_LEN)


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


class LocalSetTransformerAgg(nn.Module):
    """Per-atom local set-transformer aggregation ("idea A").

    Replaces the weighted-sum message with a small multi-head transformer over the
    atom's coordination shell. The M neighbor slots become tokens ``[h_j ‖ e_ij]``
    that do self-attention; the attention logit between two neighbors is BIASED by
    the bond angle between them (a per-head function of an RBF expansion of cos θ,
    read from the precomputed (M, M) angle matrix) — so atoms (tokens), edges (token
    features) and angles (attention bias) are fused in one operation. A learned
    center query then reads out the atom's message. Drop-in for ``aggregate``: same
    (N, atom_fea_len) output, same residual wrapper.

    The angle matrix uses sentinel 2.0 for "no angle for this pair" (padding, or a
    pair with no stored triplet); those entries contribute no bias.
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, n_heads: int = 4):
        super().__init__()
        if atom_fea_len % n_heads != 0:
            raise ValueError(
                f"atom_fea_len ({atom_fea_len}) must be divisible by "
                f"set_transformer_heads ({n_heads}).")
        self.d = atom_fea_len
        self.h = n_heads
        self.dh = atom_fea_len // n_heads
        self.tok_proj = nn.Linear(atom_fea_len + nbr_fea_len, atom_fea_len)
        self.q = nn.Linear(atom_fea_len, atom_fea_len)
        self.k = nn.Linear(atom_fea_len, atom_fea_len)
        self.v = nn.Linear(atom_fea_len, atom_fea_len)
        self.center_q = nn.Linear(atom_fea_len, atom_fea_len)
        self.out = nn.Linear(atom_fea_len, atom_fea_len)
        # RBF basis over cos θ ∈ [-1, 1]: the SAME basis MPNNData uses for the
        # edge angle vectors (one shared definition — retuning it there retunes
        # both encodings), mapped per-head to an additive attention bias.
        centers = torch.from_numpy(ANGLE_RBF_CENTERS.copy())
        self.register_buffer("angle_centers", centers)
        self.angle_width = float(centers[1] - centers[0])
        self.angle_bias = nn.Linear(ANGLE_FEA_LEN, n_heads)

    def _angle_rbf(self, cos: torch.Tensor) -> torch.Tensor:
        # cos (N, M, M) -> (N, M, M, K)
        diff = cos.unsqueeze(-1) - self.angle_centers
        return torch.exp(-(diff ** 2) / (2.0 * self.angle_width ** 2))

    def forward(self, atom_in_fea, nbr_fea_norm, nbr_fea_idx, pad_mask, nbr_angle):
        N, M = nbr_fea_idx.shape
        h, dh = self.h, self.dh
        # Neighbor tokens [h_j ‖ e_ij]  -> (N, M, d)
        nbr_h = atom_in_fea[nbr_fea_idx.reshape(-1)].view(N, M, self.d)
        tok = self.tok_proj(torch.cat([nbr_h, nbr_fea_norm], dim=2))

        def heads(x):
            return x.view(N, M, h, dh).transpose(1, 2)  # (N, h, M, dh)

        q, k, v = heads(self.q(tok)), heads(self.k(tok)), heads(self.v(tok))
        logits = (q @ k.transpose(-1, -2)) / (dh ** 0.5)  # (N, h, M, M)

        # Angle bias: only for pairs that have a real stored angle (|cos| <= 1).
        if nbr_angle is not None and nbr_angle.numel() and nbr_angle.shape[-1] == M:
            real = (nbr_angle.abs() <= 1.0).unsqueeze(1)  # (N, 1, M, M)
            bias = self.angle_bias(self._angle_rbf(nbr_angle.clamp(-1.0, 1.0)))
            logits = logits + bias.permute(0, 3, 1, 2) * real

        # Mask padded KEY slots out of the softmax.
        logits = logits.masked_fill(~pad_mask[:, None, None, :], -1e9)
        ctx = torch.softmax(logits, dim=-1) @ v  # (N, h, M, dh)

        # Center-query readout over the (real) token contexts.
        cq = self.center_q(atom_in_fea).view(N, h, 1, dh)
        r_logits = (cq @ ctx.transpose(-1, -2)).squeeze(2) / (dh ** 0.5)  # (N, h, M)
        r_logits = r_logits.masked_fill(~pad_mask[:, None, :], -1e9)
        read = (torch.softmax(r_logits, dim=-1).unsqueeze(-1) * ctx).sum(2)  # (N, h, dh)
        out = self.out(read.reshape(N, self.d))

        # Atoms with zero real neighbors: softmax over an all-masked row would be
        # uniform over padding; zero their message so they contribute nothing.
        return out * pad_mask.any(dim=1, keepdim=True).float()


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
                 ecn_weight_idx: int = ECN_WEIGHT_SRC_IDX,
                 use_coord_magnitude: bool = False, n_heads: int = 4):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.aggregation = aggregation
        self.ecn_weight_idx = ecn_weight_idx
        self.use_coord_magnitude = use_coord_magnitude

        # The set-transformer aggregation uses its own neighbor-token attention
        # (LocalSetTransformerAgg); the ecn_weighted / attention variants use the
        # EdgeNet message. Build only what the chosen variant needs.
        if aggregation == 'set_transformer':
            self.set_tr = LocalSetTransformerAgg(atom_fea_len, nbr_fea_len, n_heads)
        else:
            self.edge_net = EdgeNet(atom_fea_len, nbr_fea_len, edge_hidden_dim)
        self.norm_out = nn.LayerNorm(atom_fea_len)
        self.act = nn.Softplus()

        # The weighted aggregation normalizes its weights to sum to 1, so it is
        # intensive and throws away the atom's TOTAL coordination strength (sum of
        # raw ECoN/shared_count weights over its real neighbors) — a physically
        # meaningful signal (under- vs over-coordinated sites). When enabled, we
        # re-inject log1p(total) through a learned projection so the message keeps
        # a coordination-magnitude channel alongside the intensive average.
        if use_coord_magnitude:
            self.coord_proj = nn.Linear(1, atom_fea_len)

        # Per-feature input standardization for the EdgeNet, applied ONLY to the
        # raw edge features fed into the MLP. Default identity; filled from the
        # training set via set_edge_stats(). Stored as buffers so they save in
        # the checkpoint and apply identically at train/val/test/inference.
        self.register_buffer("edge_mean", torch.zeros(nbr_fea_len))
        self.register_buffer("edge_std", torch.ones(nbr_fea_len))

        if aggregation == 'attention':
            self.attn_fc = nn.Linear(atom_fea_len, 1)

    def set_edge_stats(self, stats):
        mean, std = stats
        self.edge_mean.copy_(torch.as_tensor(mean, dtype=self.edge_mean.dtype))
        self.edge_std.copy_(torch.as_tensor(std, dtype=self.edge_std.dtype))

    def aggregate(self, atom_in_fea: torch.Tensor,
                  nbr_fea: torch.Tensor,
                  nbr_fea_idx: torch.LongTensor,
                  nbr_angle: torch.Tensor = None) -> torch.Tensor:
        """Compute the aggregated edge message for each atom (pre-residual).

        Separated from ``forward`` so that a multi-edge-type layer can sum
        messages from several edge types before applying a single residual
        update.

        Parameters
        ----------
        atom_in_fea  : (N, atom_fea_len)
        nbr_fea      : (N, M, nbr_fea_len)
        nbr_fea_idx  : (N, M)  — local batch atom indices, padded with self-index
        nbr_angle    : (N, M, M) cos-angle matrix for 'set_transformer' (else None)

        Returns
        -------
        aggregated   : (N, atom_fea_len)
        """
        N, M = nbr_fea_idx.shape

        # Standardize edge features for the message input ONLY. The padding mask and
        # the aggregation weight column below read the RAW nbr_fea, so this does
        # not disturb padding detection or the physical ECoN/shared_count weights.
        nbr_fea_norm = (nbr_fea - self.edge_mean) / self.edge_std
        # Padding mask: real edges have at least one non-zero feature value
        pad_mask = (nbr_fea.abs().sum(dim=2) > 0)  # (N, M) bool

        if self.aggregation == 'set_transformer':
            aggregated = self.set_tr(atom_in_fea, nbr_fea_norm, nbr_fea_idx,
                                     pad_mask, nbr_angle)
        else:
            # Gather neighbor + expand center -> edge MLP input [center||nbr||edge]
            nbr_atom_fea = atom_in_fea[nbr_fea_idx.view(-1)].view(N, M, self.atom_fea_len)
            center_fea = atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len)
            edge_input = torch.cat([center_fea, nbr_atom_fea, nbr_fea_norm], dim=2)
            edge_repr = self.edge_net(edge_input)  # (N, M, atom_fea_len)

            if self.aggregation == 'ecn_weighted':
                # Raw weight column (ECoN weight for bonds, shared_count for poly).
                ecn_w = nbr_fea[:, :, self.ecn_weight_idx].clamp(min=0.0)
                ecn_w = ecn_w * pad_mask.float()
                ecn_w_sum = ecn_w.sum(dim=1, keepdim=True).clamp(min=1e-8)
                norm_w = ecn_w / ecn_w_sum  # (N, M)
                aggregated = (norm_w.unsqueeze(2) * edge_repr).sum(dim=1)

            elif self.aggregation == 'attention':
                scores = self.attn_fc(edge_repr)
                scores = scores.masked_fill(~pad_mask.unsqueeze(2), -1e9)
                attn_w = torch.softmax(scores, dim=1)  # (N, M, 1)
                aggregated = (attn_w * edge_repr).sum(dim=1)

            else:
                raise ValueError(f"Unknown aggregation: '{self.aggregation}'. "
                                 "Choose 'ecn_weighted', 'attention' or 'set_transformer'.")

        if self.use_coord_magnitude:
            # Total coordination strength = sum of the raw physical weight column
            # over real neighbors (idx is ecn_weight_center for bonds /
            # shared_count for poly). Computed from raw nbr_fea so it is independent
            # of the aggregation mode (attention discards these weights otherwise).
            coord_mag = (nbr_fea[:, :, self.ecn_weight_idx].clamp(min=0.0)
                         * pad_mask.float()).sum(dim=1, keepdim=True)  # (N, 1)
            aggregated = aggregated + self.coord_proj(torch.log1p(coord_mag))

        return aggregated

    def forward(self, atom_in_fea: torch.Tensor,
                nbr_fea: torch.Tensor,
                nbr_fea_idx: torch.LongTensor,
                nbr_angle: torch.Tensor = None) -> torch.Tensor:
        """Single-edge-type message passing: aggregate + residual update."""
        aggregated = self.aggregate(atom_in_fea, nbr_fea, nbr_fea_idx, nbr_angle)
        # Residual connection + layer norm + activation
        out = self.act(self.norm_out(aggregated + atom_in_fea))
        return out


class DualMPNNConvLayer(nn.Module):
    """Message-passing layer over two distinct edge types: bonding edges and
    polyhedral (corner/edge/face-sharing) edges.

    Each edge type has its own EdgeNet + aggregation (a separate
    ``MPNNConvLayer``), so the two relations are learned independently. Their
    aggregated messages are summed and applied with a single shared residual
    update, mirroring the single-edge ``MPNNConvLayer`` so the rest of the
    network is unchanged.

    Parameters
    ----------
    atom_fea_len    : int
    nbr_fea_len     : int   raw bonding-edge feature dimension
    poly_fea_len    : int   raw polyhedral-edge feature dimension
    edge_hidden_dim : int   hidden dimension inside each EdgeNet
    aggregation     : str   'ecn_weighted' | 'attention' | 'set_transformer'
    poly_fusion     : str   how the bond and poly messages combine:
                            'sum'  — plain addition (original behaviour); the model
                                     cannot distinguish the two relation types
                            'gate' — per-channel sigmoid gates on each message,
                                     conditioned on both messages, so the model
                                     weighs bonding vs polyhedral context per atom
    """

    def __init__(self, atom_fea_len: int, nbr_fea_len: int, poly_fea_len: int,
                 edge_hidden_dim: int, aggregation: str = 'ecn_weighted',
                 use_coord_magnitude: bool = False, n_heads: int = 4,
                 poly_fusion: str = 'sum'):
        super().__init__()
        if poly_fusion not in ('sum', 'gate'):
            raise ValueError(f"Unknown poly_fusion '{poly_fusion}' (use 'sum' or 'gate').")
        self.poly_fusion = poly_fusion
        self.bond_conv = MPNNConvLayer(
            atom_fea_len, nbr_fea_len, edge_hidden_dim, aggregation,
            ecn_weight_idx=ECN_WEIGHT_SRC_IDX,
            use_coord_magnitude=use_coord_magnitude, n_heads=n_heads,
        )
        # Bond-only set-transformer: the polyhedral channel has no bond-angle
        # matrix, so it stays on the standard weighted aggregation (ecn_weighted)
        # rather than the set-transformer. (Revisiting how poly edges are used is a
        # separate planned step.)
        poly_aggregation = 'ecn_weighted' if aggregation == 'set_transformer' else aggregation
        self.poly_conv = MPNNConvLayer(
            atom_fea_len, poly_fea_len, edge_hidden_dim, poly_aggregation,
            ecn_weight_idx=POLY_WEIGHT_IDX,
            use_coord_magnitude=use_coord_magnitude,
        )
        self.norm_out = nn.LayerNorm(atom_fea_len)
        self.act = nn.Softplus()

        if poly_fusion == 'gate':
            # One linear over [bond_msg ‖ poly_msg] emits BOTH per-channel gates
            # (chunked). Bias starts at +2 (sigmoid ≈ 0.88) so training begins
            # near the additive baseline and learns selectivity from there,
            # instead of starting half-attenuated (sigmoid(0) = 0.5).
            self.fusion_gate = nn.Linear(2 * atom_fea_len, 2 * atom_fea_len)
            nn.init.zeros_(self.fusion_gate.weight)
            nn.init.constant_(self.fusion_gate.bias, 2.0)

    def forward(self, atom_in_fea: torch.Tensor,
                nbr_fea: torch.Tensor, nbr_fea_idx: torch.LongTensor,
                poly_fea: torch.Tensor, poly_fea_idx: torch.LongTensor,
                nbr_angle: torch.Tensor = None) -> torch.Tensor:
        bond_msg = self.bond_conv.aggregate(atom_in_fea, nbr_fea, nbr_fea_idx, nbr_angle)
        poly_msg = self.poly_conv.aggregate(atom_in_fea, poly_fea, poly_fea_idx)
        if self.poly_fusion == 'gate':
            g = torch.sigmoid(self.fusion_gate(torch.cat([bond_msg, poly_msg], dim=1)))
            g_bond, g_poly = g.chunk(2, dim=1)
            fused = g_bond * bond_msg + g_poly * poly_msg
        else:
            fused = bond_msg + poly_msg
        out = self.act(self.norm_out(fused + atom_in_fea))
        return out


class CrystalMPNN(nn.Module):
    """Crystal graph neural network with learned edge representations.

    Replaces the original CGCNN ConvLayer with MPNNConvLayer (EdgeNet + aggregation).
    Architecture otherwise mirrors CrystalGraphConvNet for direct comparison.

    Parameters
    ----------
    orig_atom_fea_len : int   raw node feature dimension from the dataset
    nbr_fea_len       : int   raw bonding-edge feature dimension from the dataset
    poly_fea_len      : int   raw polyhedral-edge feature dimension from the dataset
    atom_fea_len      : int   hidden atom feature dimension (post-embedding)
    edge_hidden_dim   : int   hidden dimension inside EdgeNet
    n_conv            : int   number of message-passing layers
    h_fea_len         : int   MLP hidden dimension after global pooling
    n_h               : int   number of MLP layers after pooling
    edge_aggregation  : str   how edge messages are aggregated onto each atom:
                              'ecn_weighted' | 'attention'
    atom_pooling      : str   how atom embeddings are read out into a single
                              crystal vector:
                                'mean'      — global mean (original behaviour)
                                'mean_max'  — concat(mean, max), 2x width; max
                                              surfaces the single most active atom
                                'attention' — learned per-atom softmax weighting
                                              (single-step gated readout)
                                'set2set'   — Set2Set (Vinyals 2015): an LSTM-
                                              driven attention readout iterated
                                              for `set2set_steps` steps, 2x width
    set2set_steps     : int   number of Set2Set processing steps (only used when
                              atom_pooling == 'set2set').
    use_poly_edges    : bool  message-pass over polyhedral edges in addition to
                              bonding edges (DualMPNNConvLayer). When False,
                              poly inputs are ignored (bonding edges only).
    """

    _POOLINGS = ("mean", "mean_max", "attention", "set2set")

    def __init__(self, orig_atom_fea_len: int, nbr_fea_len: int,
                 poly_fea_len: int = 7,
                 atom_fea_len: int = 64, edge_hidden_dim: int = 128,
                 n_conv: int = 3, h_fea_len: int = 128, n_h: int = 1,
                 edge_aggregation: str = 'ecn_weighted', classification: bool = False,
                 use_poly_edges: bool = True, atom_pooling: str = 'mean',
                 set2set_steps: int = 3, dropout: float = 0.0,
                 use_coord_magnitude: bool = False, set_transformer_heads: int = 4,
                 poly_fusion: str = 'sum'):
        super().__init__()
        self.uses_angle_bias = (edge_aggregation == 'set_transformer')

        if atom_pooling not in self._POOLINGS:
            raise ValueError(f"Unknown atom_pooling '{atom_pooling}'. "
                             f"Choose one of {self._POOLINGS}.")

        self.classification = classification
        self.use_poly_edges = use_poly_edges
        self.atom_pooling = atom_pooling
        self.set2set_steps = set2set_steps
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Per-feature node-input standardization (default identity; filled from
        # the training set via set_feature_stats). Buffer -> saved in checkpoint.
        self.register_buffer("node_mean", torch.zeros(orig_atom_fea_len))
        self.register_buffer("node_std", torch.ones(orig_atom_fea_len))

        if use_poly_edges:
            self.convs = nn.ModuleList([
                DualMPNNConvLayer(atom_fea_len, nbr_fea_len, poly_fea_len,
                                  edge_hidden_dim, edge_aggregation,
                                  use_coord_magnitude=use_coord_magnitude,
                                  n_heads=set_transformer_heads,
                                  poly_fusion=poly_fusion)
                for _ in range(n_conv)
            ])
        else:
            self.convs = nn.ModuleList([
                MPNNConvLayer(atom_fea_len, nbr_fea_len, edge_hidden_dim, edge_aggregation,
                              use_coord_magnitude=use_coord_magnitude,
                              n_heads=set_transformer_heads)
                for _ in range(n_conv)
            ])

        # 'attention' learns a scalar score per atom; 'set2set' runs an LSTM
        # whose input is the previous step's 2*d readout and whose hidden state
        # (dim d) is the attention query. Both 'mean_max' and 'set2set' produce
        # a 2*d crystal vector fed to the post-pool MLP.
        if atom_pooling == 'attention':
            self.pool_attn = nn.Linear(atom_fea_len, 1)
        elif atom_pooling == 'set2set':
            self.s2s_lstm = nn.LSTM(2 * atom_fea_len, atom_fea_len, num_layers=1)
        pool_out_len = atom_fea_len * (2 if atom_pooling in ('mean_max', 'set2set') else 1)

        self.conv_to_fc = nn.Linear(pool_out_len, h_fea_len)
        self.conv_to_fc_act = nn.Softplus()

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.fc_acts = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])

        # 2-class log-softmax head for the SC/non-SC classifier, else scalar regressor.
        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)
        # Dropout on the pooled crystal vector before the readout MLP. Applied to
        # BOTH tasks now (previously classification-only). p=0.0 is a no-op, so it
        # is off unless a config sets `dropout`. Regularizes a high-capacity model
        # that otherwise memorizes the training set (large train/val MAE gap).
        self.dropout = nn.Dropout(dropout)
        if classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
        self.n_h = n_h

    def set_feature_stats(self, node, nbr, poly=None):
        """Install per-feature input standardization computed on the training set.

        node/nbr/poly are each (mean, std) arrays. nbr stats go to every bonding
        conv; poly stats to every polyhedral conv (when poly edges are used).
        """
        self.node_mean.copy_(torch.as_tensor(node[0], dtype=self.node_mean.dtype))
        self.node_std.copy_(torch.as_tensor(node[1], dtype=self.node_std.dtype))
        for conv in self.convs:
            if isinstance(conv, DualMPNNConvLayer):
                conv.bond_conv.set_edge_stats(nbr)
                if poly is not None:
                    conv.poly_conv.set_edge_stats(poly)
            else:
                conv.set_edge_stats(nbr)

    def forward(self, atom_fea: torch.Tensor,
                nbr_fea: torch.Tensor,
                nbr_fea_idx: torch.LongTensor,
                poly_fea: torch.Tensor,
                poly_fea_idx: torch.LongTensor,
                nbr_angle: torch.Tensor,
                crystal_seg: torch.LongTensor,
                n_crystals: int) -> torch.Tensor:
        atom_fea = (atom_fea - self.node_mean) / self.node_std
        atom_fea = self.embedding(atom_fea)

        # Angle bias is only consumed by the set-transformer aggregation; pass None
        # otherwise so the standard convs ignore it.
        angle = nbr_angle if self.uses_angle_bias else None
        for conv in self.convs:
            if self.use_poly_edges:
                atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx, angle)
            else:
                atom_fea = conv(atom_fea, nbr_fea, nbr_fea_idx, angle)

        crys_fea = self._pooling(atom_fea, crystal_seg, n_crystals)

        crys_fea = self.conv_to_fc_act(self.conv_to_fc(crys_fea))

        crys_fea = self.dropout(crys_fea)

        if self.n_h > 1:
            for fc, act in zip(self.fcs, self.fc_acts):
                crys_fea = act(fc(crys_fea))

        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return out

    def _pooling(self, atom_fea: torch.Tensor, seg: torch.LongTensor,
                 B: int) -> torch.Tensor:
        """Read out per-atom embeddings into one vector per crystal.

        Fully scatter-based over the segment vector ``seg`` (atom -> crystal id),
        with ``B`` crystals: no per-crystal Python loop or per-crystal kernels —
        the loop version cost ~128 small kernel launches per batch in the readout
        alone. Sums use index_add_, whose accumulation order differs from the old
        per-crystal mean, so results match to float32 rounding (~1e-7), not
        bitwise.

        Mode is set by ``atom_pooling``:
          'mean'      — average atom embedding (intensive; original behaviour).
          'mean_max'  — concat(mean, elementwise-max); max surfaces the single
                        most active atom (e.g. a superconductivity-relevant site)
                        that a mean would dilute. Output width is 2x.
          'attention' — learned scalar score per atom, softmax-normalised within
                        each crystal, then weighted sum (gated readout).
          'set2set'   — LSTM-driven multi-step attention readout (Vinyals 2015).
        """
        assert seg.shape[0] == atom_fea.shape[0]
        if self.atom_pooling == 'set2set':
            return self._set2set(atom_fea, seg, B)
        N, d = atom_fea.shape
        counts = torch.bincount(seg, minlength=B).clamp(min=1).unsqueeze(1)  # (B,1)

        def seg_mean(x):
            return x.new_zeros(B, d).index_add_(0, seg, x) / counts

        if self.atom_pooling == 'mean':
            return seg_mean(atom_fea)
        if self.atom_pooling == 'mean_max':
            mx = atom_fea.new_full((B, d), float("-inf")).scatter_reduce_(
                0, seg.unsqueeze(1).expand(N, d), atom_fea,
                reduce="amax", include_self=True)
            return torch.cat([seg_mean(atom_fea), mx], dim=1)
        # 'attention': numerically-stable per-crystal softmax over atom scores.
        e = self.pool_attn(atom_fea)                                  # (N, 1)
        seg_max = atom_fea.new_full((B, 1), float("-inf")).scatter_reduce_(
            0, seg.unsqueeze(1), e, reduce="amax", include_self=True)
        e_exp = (e - seg_max[seg]).exp()
        seg_sum = atom_fea.new_zeros(B, 1).index_add_(0, seg, e_exp)
        w = e_exp / seg_sum[seg]                                      # (N, 1)
        return atom_fea.new_zeros(B, d).index_add_(0, seg, w * atom_fea)

    def _set2set(self, atom_fea: torch.Tensor, batch: torch.LongTensor,
                 B: int) -> torch.Tensor:
        """Set2Set readout (Vinyals et al. 2015), batched over crystals.

        Per step t: query q_t = LSTM(q*_{t-1}); attention a_i = softmax_i(h_i·q_t)
        within each crystal; readout r_t = sum_i a_i h_i; q*_t = [q_t, r_t].
        Returns q*_T of width 2*atom_fea_len, one row per crystal. ``batch`` is
        the atom->crystal segment vector (already built by collate).
        """
        N, d = atom_fea.shape

        h = (atom_fea.new_zeros(1, B, d), atom_fea.new_zeros(1, B, d))
        q_star = atom_fea.new_zeros(B, 2 * d)
        for _ in range(self.set2set_steps):
            q, h = self.s2s_lstm(q_star.unsqueeze(0), h)     # q: (1, B, d)
            q = q.squeeze(0)                                  # (B, d)
            e = (atom_fea * q[batch]).sum(dim=1, keepdim=True)  # (N, 1) attention logits
            # Numerically-stable per-crystal softmax over atoms.
            seg_max = atom_fea.new_full((B, 1), float("-inf"))
            seg_max.scatter_reduce_(0, batch.unsqueeze(1), e, reduce="amax", include_self=True)
            e_exp = (e - seg_max[batch]).exp()
            seg_sum = atom_fea.new_zeros(B, 1).index_add_(0, batch, e_exp)
            a = e_exp / (seg_sum[batch] + 1e-16)              # (N, 1)
            r = atom_fea.new_zeros(B, d).index_add_(0, batch, a * atom_fea)  # (B, d)
            q_star = torch.cat([q, r], dim=1)                 # (B, 2d)
        return q_star
