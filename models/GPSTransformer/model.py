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

import warnings

import torch
import torch.nn as nn

# Neutral data constants (the angle-RBF basis shared by the featurization); not
# model code, so importing from common keeps GPS independent of MPNN.
from data import ANGLE_RBF_CENTERS, ANGLE_FEA_LEN


class ShellAttention(nn.Module):
    """Per-atom aggregation over a neighbor shell. Neighbor tokens are [h_j ‖ e_ij].

    aggregation="attention" (default): multi-head self-attention among the shell
    tokens, optionally biased by the inter-neighbor bond angle (a per-head RBF(cos θ)
    map), with a learned center query reading the attended shell into the atom's
    message. `angle=None` / `use_angle=False` drops the angle bias.

    aggregation="mean": the non-attention baseline (an ablation rung) — project the
    neighbor tokens and mean-pool over the real neighbors. No q/k/v, no angle.
    """

    def __init__(self, d: int, edge_dim: int, n_heads: int = 4, use_angle: bool = True,
                 dropout: float = 0.0, aggregation: str = "attention"):
        super().__init__()
        if aggregation not in ("attention", "mean"):
            raise ValueError("ShellAttention aggregation must be 'attention' or 'mean'.")
        self.aggregation = aggregation
        self.d = d
        # Shared by both modes: token projection of [h_j ‖ e_ij], output projection,
        # and dropout (the local channel previously had none while the global/FFN did).
        self.tok_proj = nn.Linear(d + edge_dim, d)
        self.out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)
        self.use_angle = use_angle if aggregation == "attention" else False
        if aggregation == "attention":
            if d % n_heads != 0:
                raise ValueError(f"atom_feat_len ({d}) must be divisible by heads ({n_heads}).")
            self.h, self.dh = n_heads, d // n_heads
            self.q = nn.Linear(d, d)
            self.k = nn.Linear(d, d)
            self.v = nn.Linear(d, d)
            self.center_q = nn.Linear(d, d)
            if use_angle:
                self.register_buffer("angle_centers", torch.from_numpy(ANGLE_RBF_CENTERS.copy()))
                self.angle_width = float(self.angle_centers[1] - self.angle_centers[0])
                self.angle_bias = nn.Linear(ANGLE_FEA_LEN, n_heads)
                # Warn (once) if a use_angle shell ever runs without a usable angle
                # tensor: the angle bias is the benchmark-winning component, and
                # silently dropping it (shape/config mismatch) would degrade quality.
                self._warned_no_angle = False

    def _angle_rbf(self, cos):
        diff = cos.unsqueeze(-1) - self.angle_centers
        return torch.exp(-(diff ** 2) / (2.0 * self.angle_width ** 2))

    def _forward_mean(self, h, nbr_fea_norm, nbr_fea_idx, pad_mask):
        # Non-attention baseline: project [h_j ‖ e_ij] and mean-pool over the real
        # neighbors. Atoms with no real neighbors get a zero message (identity).
        N, M = nbr_fea_idx.shape
        nbr_h = h[nbr_fea_idx.reshape(-1)].view(N, M, self.d)
        tok = self.dropout(self.tok_proj(torch.cat([nbr_h, nbr_fea_norm], dim=2)))
        mask = pad_mask.unsqueeze(-1).to(tok.dtype)              # (N, M, 1)
        msg = (tok * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)  # (N, d)
        out = self.dropout(self.out(msg))
        return out * pad_mask.any(dim=1, keepdim=True).to(out.dtype)

    def forward(self, h, nbr_fea_norm, nbr_fea_idx, pad_mask, angle):
        if self.aggregation == "mean":
            return self._forward_mean(h, nbr_fea_norm, nbr_fea_idx, pad_mask)
        N, M = nbr_fea_idx.shape
        nh, dh = self.h, self.dh
        nbr_h = h[nbr_fea_idx.reshape(-1)].view(N, M, self.d)
        tok = self.tok_proj(torch.cat([nbr_h, nbr_fea_norm], dim=2))

        def heads(x):
            return x.view(N, M, nh, dh).transpose(1, 2)  # (N, h, M, dh)

        q, k, v = heads(self.q(tok)), heads(self.k(tok)), heads(self.v(tok))
        logits = (q @ k.transpose(-1, -2)) / (dh ** 0.5)  # (N, h, M, M)

        if self.use_angle:
            if angle is not None and angle.numel() and angle.shape[-1] == M:
                real = (angle.abs() <= 1.0).unsqueeze(1)
                bias = self.angle_bias(self._angle_rbf(angle.clamp(-1.0, 1.0)))
                logits = logits + bias.permute(0, 3, 1, 2) * real
            elif not self._warned_no_angle:
                got = None if angle is None else tuple(angle.shape)
                warnings.warn(
                    f"ShellAttention(use_angle=True) got an unusable angle tensor "
                    f"(shape={got}, expected last dim M={M}); the angle bias — the "
                    f"benchmark-winning component — is being SKIPPED. Check "
                    f"build_angle_bias / max_num_nbr.", RuntimeWarning, stacklevel=2)
                self._warned_no_angle = True

        # fp16-safe mask fill: a literal -1e9 overflows to -inf in half precision,
        # which NaNs the softmax of an all-padded row (and the 0-gate below can't
        # rescue 0*NaN). dtype.min stays finite.
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~pad_mask[:, None, None, :], neg)
        ctx = self.dropout(torch.softmax(logits, dim=-1)) @ v  # (N, h, M, dh)

        cq = self.center_q(h).view(N, nh, 1, dh)
        r_logits = (cq @ ctx.transpose(-1, -2)).squeeze(2) / (dh ** 0.5)
        r_logits = r_logits.masked_fill(~pad_mask[:, None, :], neg)
        read = (self.dropout(torch.softmax(r_logits, dim=-1)).unsqueeze(-1) * ctx).sum(2)
        out = self.dropout(self.out(read.reshape(N, self.d)))
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
    crystal_seg, which collate emits contiguous & non-decreasing).

    `dist_bias` (B, heads, Lmax, Lmax), when given, is an additive per-atom-pair
    PBC distance bias on the attention logits (the long-range analog of the local
    angle bias). It is folded — together with the padding mask — into the float
    `attn_mask` MultiheadAttention accepts; without it, the cheaper bool
    key_padding_mask path is used unchanged."""

    def __init__(self, d: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x, seg, B, plan, dist_bias=None):
        intra, key_pad, Lmax = plan
        h = self.norm(x)
        padded = x.new_zeros(B, Lmax, h.shape[-1])
        padded[seg, intra] = h
        if dist_bias is None:
            attended, _ = self.attn(padded, padded, padded,
                                    key_padding_mask=key_pad, need_weights=False)
        else:
            # One additive float mask (B*heads, Lmax, Lmax): the distance bias, with
            # padded KEYS set to dtype.min (fp16/bf16-safe) for all queries.
            neg = torch.finfo(padded.dtype).min
            am = dist_bias.reshape(B * self.n_heads, Lmax, Lmax).to(padded.dtype)
            am = am.masked_fill(
                key_pad.repeat_interleave(self.n_heads, dim=0).unsqueeze(1), neg)
            attended, _ = self.attn(padded, padded, padded,
                                    attn_mask=am, need_weights=False)
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
                 gps_global_heads, ffn_mult, dropout, local_transformer=True,
                 use_bond_edges=True, shell_aggregation="attention", use_angle_bias=True):
        super().__init__()
        self.use_bond = use_bond_edges
        self.use_poly = use_poly
        self.gps_global = gps_global
        self.local_transformer = local_transformer
        # The bond shell's angle term is gated by use_angle_bias (an ablation rung);
        # the poly shell never uses angles. shell_aggregation picks attention vs the
        # non-attention mean baseline for both shells.
        if use_bond_edges:
            self.bond_attn = ShellAttention(d, nbr_dim, n_heads, use_angle=use_angle_bias,
                                            dropout=dropout, aggregation=shell_aggregation)
        if use_poly:
            self.poly_attn = ShellAttention(d, poly_dim, n_heads, use_angle=False,
                                            dropout=dropout, aggregation=shell_aggregation)
        self.norm_local = nn.LayerNorm(d)
        if local_transformer:
            self.ffn_local = FeedForward(d, ffn_mult, dropout)
        else:
            self.act = nn.Softplus()
        if gps_global:
            self.global_attn = WithinCrystalAttention(d, gps_global_heads, dropout)
        self.ffn = FeedForward(d, ffn_mult, dropout)

    def _messages(self, x, nbr_norm, nbr_idx, bond_pad, angle, poly_norm, poly_idx, poly_pad):
        msgs = []
        if self.use_bond:
            msgs.append(self.bond_attn(x, nbr_norm, nbr_idx, bond_pad, angle))
        if self.use_poly:
            msgs.append(self.poly_attn(x, poly_norm, poly_idx, poly_pad, None))
        return sum(msgs) if msgs else None       # None -> no edges this block (rung a-style)

    def _local(self, h, nbr_norm, nbr_idx, bond_pad, angle, poly_norm, poly_idx, poly_pad):
        if self.local_transformer:
            # pre-LN: shell aggregation runs in normalized space and is added back to
            # the residual stream (atoms with no real neighbors get a zero message ->
            # identity preserved); the local FFN is the nonlinearity.
            hn = self.norm_local(h)
            msg = self._messages(hn, nbr_norm, nbr_idx, bond_pad, angle, poly_norm, poly_idx, poly_pad)
            return self.ffn_local(h if msg is None else h + msg)
        # Original Increment-1 fusion (post-add LN + activation, no local FFN).
        msg = self._messages(h, nbr_norm, nbr_idx, bond_pad, angle, poly_norm, poly_idx, poly_pad)
        return self.act(self.norm_local(h if msg is None else h + msg))

    def forward(self, h, nbr_norm, nbr_idx, bond_pad, angle,
                poly_norm, poly_idx, poly_pad, seg, B, plan, dist_bias=None):
        h = self._local(h, nbr_norm, nbr_idx, bond_pad, angle,
                        poly_norm, poly_idx, poly_pad)
        if self.gps_global:
            h = self.global_attn(h, seg, B, plan, dist_bias)
        return self.ffn(h)


class GPSCrystalNet(nn.Module):
    _POOLINGS = ("mean", "mean_max")

    _AGGREGATIONS = ("attention", "mean")

    def __init__(self, orig_atom_fea_len, nbr_fea_len, poly_fea_len=7,
                 atom_fea_len=128, n_conv=4, h_fea_len=128, n_h=2,
                 use_poly_edges=True, atom_pooling="mean", dropout=0.0,
                 n_heads=8, gps_global=True, gps_global_heads=8, gps_ffn_mult=2,
                 local_transformer=True, per_atom_head=True,
                 use_bond_edges=True, shell_aggregation="attention", use_angle_bias=True,
                 use_dist_bias=False, dist_cutoff=8.0, n_dist_rbf=16,
                 classification=False):
        super().__init__()
        if atom_pooling not in self._POOLINGS:
            raise ValueError(f"GPS atom_pooling must be one of {self._POOLINGS}")
        if shell_aggregation not in self._AGGREGATIONS:
            raise ValueError(f"GPS shell_aggregation must be one of {self._AGGREGATIONS}")
        if use_dist_bias and not gps_global:
            raise ValueError("use_dist_bias=True requires gps_global=True "
                             "(the distance bias is applied to the global attention).")
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
        self.per_atom_head = per_atom_head
        self.use_dist_bias = use_dist_bias
        if use_dist_bias:
            # PBC min-image distance -> Gaussian-RBF over [0, dist_cutoff] -> per-head
            # additive bias on the global attention (the long-range analog of the
            # local angle bias). Computed once per forward, shared across blocks.
            self.register_buffer("dist_centers", torch.linspace(0.0, dist_cutoff, n_dist_rbf))
            self.dist_width = float(dist_cutoff / max(n_dist_rbf - 1, 1))
            self.dist_proj = nn.Linear(n_dist_rbf, gps_global_heads)
            self._warned_no_pos = False
            self._dist_pos_ok = None   # one-time check: real positions vs a zeros pack

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
                     local_transformer=local_transformer, use_bond_edges=use_bond_edges,
                     shell_aggregation=shell_aggregation, use_angle_bias=use_angle_bias)
            for _ in range(n_conv)])

        # Per-atom energy decomposition (default): the head maps EACH atom's
        # embedding to a scalar, then those are mean-pooled to the crystal energy
        # (E = mean_i head(h_i)) — the standard MLIP readout (energy is a sum of
        # local contributions), so a high-contribution atom isn't washed out by
        # averaging embeddings before the nonlinear head. per_atom_head=False
        # restores the pool-then-head readout (which also enables mean_max pooling).
        pool_out = atom_fea_len * (2 if atom_pooling == "mean_max" else 1)
        head_in = atom_fea_len if per_atom_head else pool_out
        self.conv_to_fc = nn.Linear(head_in, h_fea_len)
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

    def _segment_mean(self, x, seg, B):
        # Mean over the atoms of each crystal: (N, F) -> (B, F). Accumulate in fp32:
        # under bf16 autocast a many-atom index_add_ drops low-order contributions
        # and biases the pooled value (notably the per-atom-energy readout).
        counts = torch.bincount(seg, minlength=B).clamp(min=1).unsqueeze(1)
        acc = x.new_zeros(B, x.shape[1], dtype=torch.float32).index_add_(0, seg, x.float())
        return (acc / counts).to(x.dtype)

    def _head(self, x):
        # Shared MLP head; applied per-atom (N, d) in the per-atom-energy readout,
        # or per-crystal (B, pool_out) in the pool-then-head readout.
        x = self.conv_to_fc_act(self.conv_to_fc(x))
        x = self.dropout(x)
        if self.n_h > 1:
            for fc, act in zip(self.fcs, self.fc_acts):
                x = act(fc(x))
        return self.fc_out(x)

    def _pool(self, h, seg, B):
        N, d = h.shape
        mean = self._segment_mean(h, seg, B)
        if self.atom_pooling == "mean":
            return mean
        mx = h.new_full((B, d), float("-inf")).scatter_reduce_(
            0, seg.unsqueeze(1).expand(N, d), h, reduce="amax", include_self=True)
        # A crystal with zero atoms (none today — collate guarantees >=1 — but guard
        # the asymmetry: the mean half is clamp(min=1)-protected, the max half isn't)
        # would keep the -inf init and NaN the downstream Linear; map it to 0.
        mx = mx.masked_fill(mx == float("-inf"), 0.0)
        return torch.cat([mean, mx], dim=1)

    def _dist_rbf(self, dist):
        diff = dist.unsqueeze(-1) - self.dist_centers
        return torch.exp(-(diff ** 2) / (2.0 * self.dist_width ** 2))

    def _distance_bias(self, frac_coords, lattice, plan, seg, B):
        # Per-crystal min-image pairwise distances -> RBF -> per-head bias, scattered
        # into the (B, Lmax, Lmax) global-attention layout. Invariant (uses distances
        # only) and periodicity-correct (fractional min image, then * lattice).
        intra, key_pad, Lmax = plan
        fr = frac_coords.new_zeros(B, Lmax, 3)
        fr[seg, intra] = frac_coords
        df = fr.unsqueeze(2) - fr.unsqueeze(1)              # (B, Lmax, Lmax, 3) frac diff
        df = df - df.round()                                # minimum image
        dc = torch.einsum("blmk,bkc->blmc", df, lattice)    # cartesian displacement
        dist = dc.norm(dim=-1)                              # (B, Lmax, Lmax)
        bias = self.dist_proj(self._dist_rbf(dist))         # (B, Lmax, Lmax, heads)
        return bias.permute(0, 3, 1, 2)                     # (B, heads, Lmax, Lmax)

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx,
                nbr_angle, crystal_seg, n_crystals, frac_coords=None, lattice=None,
                nbr_jimage=None):
        h = self.embedding((atom_fea - self.node_mean) / self.node_std)
        # Standardize edge features once (constant across blocks); pad masks read
        # the RAW features (real edges have a non-zero feature row).
        nbr_norm = (nbr_fea - self.bond_mean) / self.bond_std
        bond_pad = (nbr_fea.abs().sum(dim=2) > 0)
        poly_norm = (poly_fea - self.poly_mean) / self.poly_std
        poly_pad = (poly_fea.abs().sum(dim=2) > 0)

        # Padding layout for the global channel: computed ONCE (one host sync) and
        # shared across blocks. None when global attention is disabled or there are
        # no blocks (n_conv=0 -> raw embedded atoms straight to the head, rung a).
        plan = (segment_plan(crystal_seg, n_crystals)
                if self.gps_global and len(self.blocks) else None)

        # Long-range distance bias: computed ONCE (positions are constant across
        # blocks) and shared. None unless enabled and positions are in the batch.
        dist_bias = None
        if self.use_dist_bias and plan is not None:
            if frac_coords is None or lattice is None:
                if not self._warned_no_pos:
                    warnings.warn(
                        "use_dist_bias=True but the batch carries no positions "
                        "(frac_coords/lattice). Use the GPS geometry collate; "
                        "skipping the distance bias.", RuntimeWarning)
                    self._warned_no_pos = True
            else:
                if self._dist_pos_ok is None:
                    # One sync, once: an all-zero lattice means a positionless pack
                    # (e.g. packed_v1) where the collate fills geometry with zeros,
                    # so frac_coords/lattice are non-None but meaningless.
                    self._dist_pos_ok = bool(lattice.abs().sum().item() > 0)
                    if not self._dist_pos_ok:
                        warnings.warn(
                            "use_dist_bias=True but the batch lattice is all-zero "
                            "(positionless pack); the distance bias is disabled. Use a "
                            "positioned pack (packed_v2).", RuntimeWarning)
                if self._dist_pos_ok:
                    dist_bias = self._distance_bias(frac_coords, lattice, plan,
                                                    crystal_seg, n_crystals)

        for block in self.blocks:
            h = block(h, nbr_norm, nbr_fea_idx, bond_pad, nbr_angle,
                      poly_norm, poly_fea_idx, poly_pad, crystal_seg, n_crystals,
                      plan, dist_bias)

        if self.per_atom_head:
            # E = mean_i head(h_i): per-atom energy, then averaged over the crystal's
            # atoms (intensive target) — keeps a high-contribution atom from being
            # washed out by pooling embeddings before the nonlinear head.
            out = self._segment_mean(self._head(h), crystal_seg, n_crystals)
        else:
            out = self._head(self._pool(h, crystal_seg, n_crystals))
        return self.logsoftmax(out) if self.classification else out
