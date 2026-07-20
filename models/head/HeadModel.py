"""T_c head: the small trainable model that sits on a frozen, pluggable encoder.

Architecture (decided 2026-06-12, parameter-budget-first):

    pooled encoder embedding (2*D_enc) --PCA/whiten (buffers, 0 params)--> k
    physical descriptors (P)         --standardize (buffers)-----------> P
            [k || P] -> LayerNorm -> Linear(k+P, h) -> Softplus -> Dropout
                          |-> class head Linear(h, 2)   (SC / non-SC logits)
                          `-> t_c head  Linear(h, 1)    (z-space, log1p Kelvin)

Everything data-dependent but non-trainable (PCA basis, feature standardizers,
target normalizer) lives in buffers so a checkpoint is self-contained. Fresh
trainable parameters are ~7.5k at the default k=64, h=64 — inside the ~10k
budget for ~5.8k noisy T_c labels. The encoder never appears here: it
contributed the pooled embedding offline (see embed_mace.py), which is what
makes encoders interchangeable beneath this head.

Pooling (`pooling`, decided 2026-06-29): how the per-atom embedding (N_atoms,D)
collapses to one structure vector before the trunk.
  - "meanmax"   : parameter-free [mean||max], pooled offline -> PCA/whiten (above).
  - "deepsets"  : [mean||max] of a learned per-atom map g(h_i); mirrors the
                  encoder's pretraining readout E = mean_i head(h_i), so the
                  nonlinearity acts per-atom BEFORE aggregation (Jensen).
  - "attention" : gated attention-MIL pool — a learned query weights atoms so the
                  head can focus on the active sublattice (localized SC physics),
                  concatenated with a plain value-mean for the global view.
The learned pools consume the raw per-atom embedding (segment/CSR-packed, see
HeadData), add fresh params (reported), and replace the PCA step.
"""

import torch
import torch.nn as nn


def _segment_mean(x, seg, n):
    """Mean of rows of x (A,D) grouped by segment id seg (A,) into n groups."""
    out = torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device)
    out.index_add_(0, seg, x)
    cnt = torch.zeros(n, 1, dtype=x.dtype, device=x.device)
    cnt.index_add_(0, seg, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device))
    return out / cnt.clamp(min=1.0)


def _segment_max(x, seg, n):
    """Per-feature max of x (A,D) grouped by seg into n groups (0 for empties)."""
    out = torch.full((n, x.shape[1]), float("-inf"), dtype=x.dtype, device=x.device)
    out.scatter_reduce_(0, seg.unsqueeze(1).expand_as(x), x, reduce="amax",
                        include_self=True)
    return out.masked_fill(out == float("-inf"), 0.0)


def _segment_softmax(scores, seg, n):
    """Softmax of per-atom scores (A,) within each of n segments -> weights (A,)."""
    m = torch.full((n,), float("-inf"), dtype=scores.dtype, device=scores.device)
    m.scatter_reduce_(0, seg, scores, reduce="amax", include_self=True)
    e = (scores - m[seg]).exp()
    s = torch.zeros(n, dtype=scores.dtype, device=scores.device)
    s.index_add_(0, seg, e)
    return e / s[seg].clamp(min=1e-12)


class DeepSetsPool(nn.Module):
    """pooled = [mean_i g(h_i) || max_i g(h_i)] — a per-atom map THEN aggregate,
    matching the encoder's pretraining readout (mean of a per-atom head). The
    nonlinearity acts before pooling, which plain mean||max cannot express.

    Pooling-size study knobs (small-sample regime):
    - ``rank``: LOW-RANK g (in->rank->pool_dim) decouples capacity from output
      width — e.g. 128->16->64 is ~3.1k params vs the full 128->64's ~8.3k, at
      the SAME pooled width. None -> the full single Linear (default, unchanged).
    - ``agg``: which aggregations to concat — "meanmax" (default, out=2*pool_dim),
      "mean" or "max" (out=pool_dim). Tests where the (cuprate) signal lives: the
      max branch captures the single most extreme site (the active sublattice)."""

    def __init__(self, in_dim: int, pool_dim: int = 32, rank: int = None, agg: str = "meanmax"):
        super().__init__()
        if rank:
            self.g = nn.Sequential(nn.Linear(in_dim, int(rank)), nn.Softplus(),
                                   nn.Linear(int(rank), pool_dim))
        else:
            self.g = nn.Sequential(nn.Linear(in_dim, pool_dim))
        self.act = nn.Softplus()
        self.agg = agg
        self.out_dim = pool_dim * (2 if agg == "meanmax" else 1)

    def forward(self, x, seg, n):
        g = self.act(self.g(x))
        if self.agg == "mean":
            return _segment_mean(g, seg, n)
        if self.agg == "max":
            return _segment_max(g, seg, n)
        return torch.cat([_segment_mean(g, seg, n), _segment_max(g, seg, n)], dim=1)


class AttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al. 2018, attention-MIL): a learned query
    scores each atom and a softmax-over-atoms weights a value projection, so the
    head can concentrate on the few SC-relevant sites. Concatenated with the
    plain value-mean so the global composition is never lost.
    pooled = [ sum_i a_i v(h_i) || mean_i v(h_i) ]."""

    def __init__(self, in_dim: int, pool_dim: int = 32, att_dim: int = None):
        super().__init__()
        att_dim = att_dim or pool_dim
        self.v = nn.Linear(in_dim, pool_dim)
        self.attn = nn.Linear(in_dim, att_dim)     # feature branch (tanh)
        self.gate = nn.Linear(in_dim, att_dim)     # gating branch (sigmoid)
        self.score = nn.Linear(att_dim, 1)
        self.out_dim = 2 * pool_dim

    def forward(self, x, seg, n):
        v = self.v(x)
        a = self.score(torch.tanh(self.attn(x)) * torch.sigmoid(self.gate(x))).squeeze(-1)
        a = _segment_softmax(a, seg, n)
        ctx = torch.zeros(n, v.shape[1], dtype=v.dtype, device=v.device)
        ctx.index_add_(0, seg, v * a.unsqueeze(1))
        return torch.cat([ctx, _segment_mean(v, seg, n)], dim=1)


class PCAWhiten(nn.Module):
    """Parameter-free linear compressor, fit once on the training matrix.

    Stores mean / principal axes / component scales as buffers; forward
    projects to k whitened (unit-variance) components.
    """

    def __init__(self, in_dim: int, k: int):
        super().__init__()
        self.k = k
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("components", torch.zeros(in_dim, k))
        self.register_buffer("inv_scale", torch.ones(k))

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        x = x.double()
        mean = x.mean(0)
        xc = x - mean
        # Economy SVD of the centered matrix: principal axes = right singular
        # vectors; component std = s / sqrt(n-1).
        _, s, vt = torch.linalg.svd(xc, full_matrices=False)
        k = min(self.k, vt.shape[0])
        comp = vt[:k].T
        scale = s[:k] / max(x.shape[0] - 1, 1) ** 0.5
        scale = torch.clamp(scale, min=1e-8)
        self.mean.copy_(mean.float())
        self.components[:, :k] = comp.float()
        self.inv_scale[:k] = (1.0 / scale).float()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) @ self.components * self.inv_scale


class Standardizer(nn.Module):
    """Per-feature (x - mean) / std with train-split statistics in buffers."""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        self.mean.copy_(x.mean(0))
        self.std.copy_(x.std(0).clamp(min=1e-8))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class TcHead(nn.Module):
    def __init__(self, enc_dim: int, phys_dim: int, pca_k: int = 64,
                 hidden: int = 64, dropout: float = 0.2,
                 pooling: str = "meanmax", pool_dim: int = 32,
                 n_classes: int = 2, head_arch: str = "concat",
                 pool_rank: int = None, pool_agg: str = "meanmax"):
        super().__init__()
        # `enc_dim` is the pooled width (2*D) for meanmax, the per-atom width (D)
        # for the learned pools — HeadMain passes the right one.
        self.pooling = pooling
        if pooling == "meanmax":
            self.pca = PCAWhiten(enc_dim, pca_k)
            pooled_dim = pca_k
        elif pooling == "deepsets":
            self.pool = DeepSetsPool(enc_dim, pool_dim, rank=pool_rank, agg=pool_agg)
            pooled_dim = self.pool.out_dim
        elif pooling == "attention":
            self.pool = AttentionPool(enc_dim, pool_dim)
            pooled_dim = self.pool.out_dim
        else:
            raise ValueError(f"unknown pooling {pooling!r}")
        self.phys_std = Standardizer(phys_dim)
        # head_arch — how the composition descriptors enter (small-sample regime):
        #   "concat"      : [pooled || phys] -> trunk (the fusion default)
        #   "struct_only" : trunk sees the structure embedding ALONE (no phys) — a
        #                   diagnostic of how much Tc signal the encoder h carries unaided
        #   "resid"       : trunk sees structure only AND a direct LINEAR composition
        #                   head is ADDED to the Tc output, so the structure MLP learns
        #                   only the RESIDUAL over composition (the wide-and-deep
        #                   regularizer — composition is what XGBoost already fits well).
        self.head_arch = head_arch
        in_dim = (pooled_dim + phys_dim) if head_arch == "concat" else pooled_dim
        if head_arch == "resid":
            self.comp_head = nn.Linear(phys_dim, 1)   # linear composition -> z
        self.norm = nn.LayerNorm(in_dim)
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.Softplus(),
                                   nn.Dropout(dropout))
        # n_classes=2: the legacy SC/non-SC pretraining head (unchanged default).
        # n_classes=4: the ground-state hurdle head — SC / FM / AFM / both, trained
        # with a MASKED cross-entropy (class -1 = unknown ground state, e.g. the
        # 3DSC tc=0 parents, contributes no gradient); the regression head then
        # trains on SC rows only and inference is E[Tc] = P(SC) * Tc_reg.
        self.n_classes = n_classes
        self.class_head = nn.Linear(hidden, n_classes)
        self.tc_head = nn.Linear(hidden, 1)
        # log1p-Kelvin z-normalization of the regression target (train split).
        self.register_buffer("tc_mean", torch.zeros(1))
        self.register_buffer("tc_std", torch.ones(1))

    @torch.no_grad()
    def fit_target(self, tc_kelvin: torch.Tensor):
        z = torch.log1p(tc_kelvin)
        self.tc_mean.copy_(z.mean().reshape(1))
        self.tc_std.copy_(z.std().clamp(min=1e-8).reshape(1))
        return self

    def target_to_z(self, tc_kelvin: torch.Tensor) -> torch.Tensor:
        return (torch.log1p(tc_kelvin) - self.tc_mean) / self.tc_std

    def z_to_kelvin(self, z: torch.Tensor) -> torch.Tensor:
        return torch.expm1(z * self.tc_std + self.tc_mean).clamp(min=0.0)

    def _pooled(self, x, seg=None, n=None):
        """meanmax: x is the offline-pooled (n,2D) vector -> PCA. learned pools:
        x is the per-atom (A,D) embedding with segment ids seg into n structures."""
        if self.pooling == "meanmax":
            return self.pca(x)
        return self.pool(x, seg, n)

    def features(self, x, phys, seg=None, n=None) -> torch.Tensor:
        pooled = self._pooled(x, seg, n)
        if self.head_arch == "concat":
            pooled = torch.cat([pooled, self.phys_std(phys)], dim=1)
        return self.trunk(self.norm(pooled))          # struct_only/resid: structure alone

    def forward(self, x, phys, seg=None, n=None):
        h = self.features(x, phys, seg, n)
        tc_z = self.tc_head(h).squeeze(-1)
        if self.head_arch == "resid":                 # + linear composition term
            tc_z = tc_z + self.comp_head(self.phys_std(phys)).squeeze(-1)
        return self.class_head(h), tc_z

    def n_fresh_params(self) -> int:
        """Trainable parameters (buffers excluded) — the budgeted quantity."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
