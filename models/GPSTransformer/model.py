"""GPSCrystalNet — a standalone hybrid local/global crystal encoder.

Increment 1 of the GPS spec (see ARCHITECTURE.md): the benchmark-winning
angle-biased local shell attention + the polyhedral channel + an interleaved
**within-crystal global attention** (GraphGPS block), energy regression,
invariant features. Fully self-contained — depends only on torch and on neutral
DATA constants (the angle-RBF basis) from models/common; it does NOT import any
MPNN model code, so GPSTransformer and MPNN are independent.

The local channel is now a *full* pre-LN transformer sublayer (clean residual
shell attention + its own FFN), matching the global channel; `local_transformer`
toggles back to the original Increment-1 fusion for ablation (see GPSBlock).

Forward signature matches the shared collate contract
(atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx, nbr_angle,
crystal_seg, n_crystals), so it runs under the shared trainer (common/train.py).

Deferred to later increments: register tokens, global->local query conditioning,
ECoN-prior logits, multi-task heads (forces/magmom/DOS), position-differentiable
forces.
"""

import torch
import torch.nn as nn

# Neutral data constants (the angle-RBF basis shared by the featurization); not
# model code, so importing from common keeps GPS independent of MPNN.
from data import ANGLE_RBF_CENTERS, ANGLE_FEA_LEN


class ShellAttention(nn.Module):
    """Per-atom multi-head self-attention over a neighbor shell, optionally biased
    by the inter-neighbor bond angle (a per-head map of an RBF(cos θ) expansion).

    Neighbor tokens are [h_j ‖ e_ij]; a learned center query reads the attended
    shell into the atom's message. `angle=None` (e.g. the polyhedral shell, which
    has no stored angle matrix) simply drops the angle bias.
    """

    def __init__(self, d: int, edge_dim: int, n_heads: int = 4, use_angle: bool = True):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"atom_feat_len ({d}) must be divisible by heads ({n_heads}).")
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.tok_proj = nn.Linear(d + edge_dim, d)
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.center_q = nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.use_angle = use_angle
        if use_angle:
            self.register_buffer("angle_centers", torch.from_numpy(ANGLE_RBF_CENTERS.copy()))
            self.angle_width = float(self.angle_centers[1] - self.angle_centers[0])
            self.angle_bias = nn.Linear(ANGLE_FEA_LEN, n_heads)

    def _angle_rbf(self, cos):
        diff = cos.unsqueeze(-1) - self.angle_centers
        return torch.exp(-(diff ** 2) / (2.0 * self.angle_width ** 2))

    def forward(self, h, nbr_fea_norm, nbr_fea_idx, pad_mask, angle):
        N, M = nbr_fea_idx.shape
        nh, dh = self.h, self.dh
        nbr_h = h[nbr_fea_idx.reshape(-1)].view(N, M, self.d)
        tok = self.tok_proj(torch.cat([nbr_h, nbr_fea_norm], dim=2))

        def heads(x):
            return x.view(N, M, nh, dh).transpose(1, 2)  # (N, h, M, dh)

        q, k, v = heads(self.q(tok)), heads(self.k(tok)), heads(self.v(tok))
        logits = (q @ k.transpose(-1, -2)) / (dh ** 0.5)  # (N, h, M, M)

        if self.use_angle and angle is not None and angle.numel() and angle.shape[-1] == M:
            real = (angle.abs() <= 1.0).unsqueeze(1)
            bias = self.angle_bias(self._angle_rbf(angle.clamp(-1.0, 1.0)))
            logits = logits + bias.permute(0, 3, 1, 2) * real

        # fp16-safe mask fill: a literal -1e9 overflows to -inf in half precision,
        # which NaNs the softmax of an all-padded row (and the 0-gate below can't
        # rescue 0*NaN). dtype.min stays finite.
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~pad_mask[:, None, None, :], neg)
        ctx = torch.softmax(logits, dim=-1) @ v  # (N, h, M, dh)

        cq = self.center_q(h).view(N, nh, 1, dh)
        r_logits = (cq @ ctx.transpose(-1, -2)).squeeze(2) / (dh ** 0.5)
        r_logits = r_logits.masked_fill(~pad_mask[:, None, :], neg)
        read = (torch.softmax(r_logits, dim=-1).unsqueeze(-1) * ctx).sum(2)
        out = self.out(read.reshape(N, self.d))
        return out * pad_mask.any(dim=1, keepdim=True).float()


def segment_plan(seg, B):
    """Per-crystal padding layout for within-crystal attention: (intra, key_pad, Lmax).

    `intra` is each atom's position within its crystal; `key_pad[b, s]` is True for
    padding slots. Computed ONCE per forward and shared across blocks — the single
    `.item()` here is the only host sync (recomputing it per block would add one
    sync per layer, undoing collate's sync-avoidance design). Relies on `seg` being
    contiguous & non-decreasing (collate: repeat_interleave(arange(B), counts)).
    """
    N = seg.shape[0]
    counts = torch.bincount(seg, minlength=B)
    Lmax = int(counts.max().item())
    offsets = counts.cumsum(0) - counts
    intra = torch.arange(N, device=seg.device) - offsets[seg]
    key_pad = torch.ones(B, Lmax, dtype=torch.bool, device=seg.device)
    key_pad[seg, intra] = False
    return intra, key_pad, Lmax


class WithinCrystalAttention(nn.Module):
    """Pre-LN residual MHSA over the atoms of each crystal (block-diagonal by
    crystal_seg, which collate emits contiguous & non-decreasing)."""

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, seg, B, plan):
        intra, key_pad, Lmax = plan
        N, d = x.shape
        h = self.norm(x)
        padded = x.new_zeros(B, Lmax, d)
        padded[seg, intra] = h
        attended, _ = self.attn(padded, padded, padded,
                                key_padding_mask=key_pad, need_weights=False)
        return x + attended[seg, intra]


class FeedForward(nn.Module):
    def __init__(self, d: int, mult: int = 2, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.net = nn.Sequential(nn.Linear(d, mult * d), nn.Softplus(),
                                 nn.Dropout(dropout), nn.Linear(mult * d, d))

    def forward(self, x):
        return x + self.net(self.norm(x))


class GPSBlock(nn.Module):
    """One GPS layer: a local transformer sublayer, then the (optional) global
    transformer sublayer, then the block-final FFN.

    local_transformer=True (default): the local channel is a *full* pre-LN
    transformer sublayer -- bond(+poly) shell attention as a clean residual
    `h <- h + Attn(LN(h))`, followed by its own pre-LN FFN. Every sublayer in the
    block is then the canonical `h <- h + Sub(LN(h))`, the local FFN supplying the
    sublayer's nonlinearity (no activation sits on the residual add).

    local_transformer=False: the original Increment-1 fusion
    `h <- Softplus(LN(h + bond(+poly)))` with no local FFN -- kept so the validated
    component can be reproduced as an ablation baseline.
    """

    def __init__(self, d, nbr_dim, poly_dim, n_heads, use_poly, gps_global,
                 gps_global_heads, ffn_mult, dropout, local_transformer=True):
        super().__init__()
        self.use_poly = use_poly
        self.gps_global = gps_global
        self.local_transformer = local_transformer
        self.bond_attn = ShellAttention(d, nbr_dim, n_heads, use_angle=True)
        if use_poly:
            self.poly_attn = ShellAttention(d, poly_dim, n_heads, use_angle=False)
        self.norm_local = nn.LayerNorm(d)
        if local_transformer:
            self.ffn_local = FeedForward(d, ffn_mult, dropout)
        else:
            self.act = nn.Softplus()
        if gps_global:
            self.global_attn = WithinCrystalAttention(d, gps_global_heads, dropout)
        self.ffn = FeedForward(d, ffn_mult, dropout)

    def _local(self, h, nbr_norm, nbr_idx, bond_pad, angle, poly_norm, poly_idx, poly_pad):
        if self.local_transformer:
            # pre-LN: the shell attention runs entirely in normalized space and is
            # added back to the residual stream (atoms with no real neighbors get a
            # zero message -> identity preserved); the local FFN is the nonlinearity.
            hn = self.norm_local(h)
            msg = self.bond_attn(hn, nbr_norm, nbr_idx, bond_pad, angle)
            if self.use_poly:
                msg = msg + self.poly_attn(hn, poly_norm, poly_idx, poly_pad, None)
            return self.ffn_local(h + msg)
        # Original Increment-1 fusion (post-add LN + activation, no local FFN).
        msg = self.bond_attn(h, nbr_norm, nbr_idx, bond_pad, angle)
        if self.use_poly:
            msg = msg + self.poly_attn(h, poly_norm, poly_idx, poly_pad, None)
        return self.act(self.norm_local(h + msg))

    def forward(self, h, nbr_norm, nbr_idx, bond_pad, angle,
                poly_norm, poly_idx, poly_pad, seg, B, plan):
        h = self._local(h, nbr_norm, nbr_idx, bond_pad, angle,
                        poly_norm, poly_idx, poly_pad)
        if self.gps_global:
            h = self.global_attn(h, seg, B, plan)
        return self.ffn(h)


class GPSCrystalNet(nn.Module):
    _POOLINGS = ("mean", "mean_max")

    def __init__(self, orig_atom_fea_len, nbr_fea_len, poly_fea_len=7,
                 atom_fea_len=128, n_conv=4, h_fea_len=128, n_h=2,
                 use_poly_edges=True, atom_pooling="mean", dropout=0.0,
                 n_heads=8, gps_global=True, gps_global_heads=8, gps_ffn_mult=2,
                 local_transformer=True, classification=False):
        super().__init__()
        if atom_pooling not in self._POOLINGS:
            raise ValueError(f"GPS atom_pooling must be one of {self._POOLINGS}")
        # Validate head divisibility up front with clear messages (ShellAttention
        # also checks n_heads; nn.MultiheadAttention's own error for global heads is
        # opaque, so check it here too).
        if atom_fea_len % n_heads != 0:
            raise ValueError(f"atom_feat_len ({atom_fea_len}) must be divisible by "
                             f"n_heads/set_transformer_heads ({n_heads}).")
        if gps_global and atom_fea_len % gps_global_heads != 0:
            raise ValueError(f"atom_feat_len ({atom_fea_len}) must be divisible by "
                             f"gps_global_heads ({gps_global_heads}).")
        self.use_poly_edges = use_poly_edges
        self.atom_pooling = atom_pooling
        self.gps_global = gps_global
        self.classification = classification
        self.n_h = n_h

        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.register_buffer("node_mean", torch.zeros(orig_atom_fea_len))
        self.register_buffer("node_std", torch.ones(orig_atom_fea_len))
        self.register_buffer("bond_mean", torch.zeros(nbr_fea_len))
        self.register_buffer("bond_std", torch.ones(nbr_fea_len))
        self.register_buffer("poly_mean", torch.zeros(poly_fea_len))
        self.register_buffer("poly_std", torch.ones(poly_fea_len))

        self.blocks = nn.ModuleList([
            GPSBlock(atom_fea_len, nbr_fea_len, poly_fea_len, n_heads, use_poly_edges,
                     gps_global, gps_global_heads, gps_ffn_mult, dropout,
                     local_transformer=local_transformer)
            for _ in range(n_conv)])

        pool_out = atom_fea_len * (2 if atom_pooling == "mean_max" else 1)
        self.conv_to_fc = nn.Linear(pool_out, h_fea_len)
        self.conv_to_fc_act = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.fc_acts = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(h_fea_len, 2 if classification else 1)
        if classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)

    def set_feature_stats(self, node, nbr, poly=None):
        self.node_mean.copy_(torch.as_tensor(node[0], dtype=self.node_mean.dtype))
        self.node_std.copy_(torch.as_tensor(node[1], dtype=self.node_std.dtype))
        self.bond_mean.copy_(torch.as_tensor(nbr[0], dtype=self.bond_mean.dtype))
        self.bond_std.copy_(torch.as_tensor(nbr[1], dtype=self.bond_std.dtype))
        if poly is not None:
            self.poly_mean.copy_(torch.as_tensor(poly[0], dtype=self.poly_mean.dtype))
            self.poly_std.copy_(torch.as_tensor(poly[1], dtype=self.poly_std.dtype))

    def _pool(self, h, seg, B):
        N, d = h.shape
        counts = torch.bincount(seg, minlength=B).clamp(min=1).unsqueeze(1)
        mean = h.new_zeros(B, d).index_add_(0, seg, h) / counts
        if self.atom_pooling == "mean":
            return mean
        mx = h.new_full((B, d), float("-inf")).scatter_reduce_(
            0, seg.unsqueeze(1).expand(N, d), h, reduce="amax", include_self=True)
        # A crystal with zero atoms (none today — collate guarantees >=1 — but guard
        # the asymmetry: the mean half is clamp(min=1)-protected, the max half isn't)
        # would keep the -inf init and NaN the downstream Linear; map it to 0.
        mx = mx.masked_fill(mx == float("-inf"), 0.0)
        return torch.cat([mean, mx], dim=1)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx,
                nbr_angle, crystal_seg, n_crystals):
        h = self.embedding((atom_fea - self.node_mean) / self.node_std)
        # Standardize edge features once (constant across blocks); pad masks read
        # the RAW features (real edges have a non-zero feature row).
        nbr_norm = (nbr_fea - self.bond_mean) / self.bond_std
        bond_pad = (nbr_fea.abs().sum(dim=2) > 0)
        poly_norm = (poly_fea - self.poly_mean) / self.poly_std
        poly_pad = (poly_fea.abs().sum(dim=2) > 0)

        # Padding layout for the global channel: computed ONCE (one host sync) and
        # shared across blocks. None when global attention is disabled.
        plan = segment_plan(crystal_seg, n_crystals) if self.gps_global else None
        for block in self.blocks:
            h = block(h, nbr_norm, nbr_fea_idx, bond_pad, nbr_angle,
                      poly_norm, poly_fea_idx, poly_pad, crystal_seg, n_crystals, plan)

        crys = self.conv_to_fc_act(self.conv_to_fc(self._pool(h, crystal_seg, n_crystals)))
        crys = self.dropout(crys)
        if self.n_h > 1:
            for fc, act in zip(self.fcs, self.fc_acts):
                crys = act(fc(crys))
        out = self.fc_out(crys)
        return self.logsoftmax(out) if self.classification else out
