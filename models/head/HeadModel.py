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
      max branch captures the single most extreme site (the active sublattice).
    - ``phi`` (2026-09-08, the phi-budget test — at 256-d the full Linear g is
      16.4k of the head's 27.9k params): what the per-atom map is.
        "linear": softplus(W h + b)              (default; W in->pool_dim, rank optional)
        "diag"  : softplus(a * h + b), a,b in R^in — an ELEMENTWISE nonlinearity
                  before pooling, no mixing (2*in params). The pooled vector is
                  then n_agg*in wide, so a FROZEN PCAWhiten (fit once on the
                  training pool, 0 params) brings it to the same n_agg*pool_dim
                  width the trunk would see under "linear".
        "pca"   : a FROZEN PCAWhiten in->pca_in on the per-atom rows (0 params),
                  then softplus(W' x + b) with W' pca_in->pool_dim (~2k at 32->64).
      Both frozen variants need ``fit(x, seg, n)`` on the training per-atom rows
      before use (``needs_fit``); TcHead.fit_pool forwards it."""

    def __init__(self, in_dim: int, pool_dim: int = 32, rank: int = None, agg: str = "meanmax",
                 phi: str = "linear", pca_in: int = 32):
        super().__init__()
        if agg not in ("mean", "max", "meanmax"):
            # Validate HERE: an unknown agg would compute out_dim=pool_dim below but
            # fall through to the meanmax concat (2*pool_dim) in forward — an opaque
            # LayerNorm shape crash deep in training instead of a named config error.
            raise ValueError(f"unknown agg {agg!r} (use 'mean', 'max' or 'meanmax')")
        if phi not in ("linear", "diag", "pca"):
            raise ValueError(f"unknown phi {phi!r} (use 'linear', 'diag' or 'pca')")
        if phi == "diag" and rank:
            raise ValueError("pool_rank applies to phi 'linear'/'pca' only")
        n_agg = 2 if agg == "meanmax" else 1
        self.phi = phi
        g_in = in_dim
        if phi == "pca":
            self.pre = PCAWhiten(in_dim, int(pca_in))   # frozen; fit() on train atoms
            g_in = int(pca_in)
        if phi == "diag":
            self.scale = nn.Parameter(torch.ones(in_dim))
            self.shift = nn.Parameter(torch.zeros(in_dim))
            self.post = PCAWhiten(in_dim * n_agg, pool_dim * n_agg)  # frozen; fit() on train pool
        elif rank:
            self.g = nn.Sequential(nn.Linear(g_in, int(rank)), nn.Softplus(),
                                   nn.Linear(int(rank), pool_dim))
        else:
            # Bare Linear (NOT a one-element Sequential): keeps the historical
            # state_dict keys (pool.g.weight, not pool.g.0.weight) so pre-existing
            # head checkpoints stay loadable key-for-key.
            self.g = nn.Linear(g_in, pool_dim)
        self.act = nn.Softplus()
        self.agg = agg
        self.out_dim = pool_dim * n_agg

    @property
    def needs_fit(self) -> bool:
        return self.phi in ("diag", "pca")

    def _agg(self, g, seg, n):
        if self.agg == "mean":
            return _segment_mean(g, seg, n)
        if self.agg == "max":
            return _segment_max(g, seg, n)
        return torch.cat([_segment_mean(g, seg, n), _segment_max(g, seg, n)], dim=1)

    @torch.no_grad()
    def fit(self, x, seg, n):
        """Fit the frozen projection(s) on the TRAINING per-atom rows x (A,D)
        with segment ids seg into n structures. No-op for phi 'linear'."""
        if self.phi == "pca":
            self.pre.fit(x)
        elif self.phi == "diag":
            self.post.fit(self._agg(self.act(x * self.scale + self.shift), seg, n))
        return self

    def forward(self, x, seg, n):
        if self.phi == "diag":
            return self.post(self._agg(self.act(x * self.scale + self.shift), seg, n))
        if self.phi == "pca":
            x = self.pre(x)
        return self._agg(self.act(self.g(x)), seg, n)


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
    """T_c regression + SC-class head over pooled encoder embeddings.

    Pipeline: pool per-atom h (pooling: "meanmax" -> PCAWhiten(pca_k);
    "deepsets"/"attention" -> learned pool sized by pool_dim, with DeepSets
    knobs pool_rank (low-rank g), pool_agg (mean/max/meanmax) and pool_phi
    (linear/diag/pca per-atom map; diag/pca need fit_pool)) -> combine
    with the standardized composition descriptors per head_arch:
      concat      — [pooled || phys] -> LayerNorm -> 1-hidden trunk (default)
      struct_only — pooled embedding alone (no descriptors; the encoder-only arm)
      resid       — linear composition baseline + structure-MLP residual
    -> tc_head (z-space regression, z = standardized log1p Kelvin) + class_head
    (n_classes; 2 = SC/non-SC, >2 = ground-state hurdle classes).
    Fresh-parameter count via n_fresh_params(); z<->Kelvin via fit_target /
    z_to_kelvin. All knobs default to the historical behavior.
    """

    def __init__(self, enc_dim: int, phys_dim: int, pca_k: int = 64,
                 hidden: int = 64, dropout: float = 0.2,
                 pooling: str = "meanmax", pool_dim: int = 32,
                 n_classes: int = 2, head_arch: str = "concat",
                 pool_rank: int = None, pool_agg: str = "meanmax",
                 pool_phi: str = "linear", pool_pca_in: int = 32):
        super().__init__()
        # `enc_dim` is the pooled width (2*D) for meanmax, the per-atom width (D)
        # for the learned pools — HeadMain passes the right one.
        self.pooling = pooling
        if pooling == "meanmax":
            self.pca = PCAWhiten(enc_dim, pca_k)
            pooled_dim = pca_k
        elif pooling == "deepsets":
            self.pool = DeepSetsPool(enc_dim, pool_dim, rank=pool_rank, agg=pool_agg,
                                     phi=pool_phi, pca_in=pool_pca_in)
            pooled_dim = self.pool.out_dim
        elif pooling == "attention":
            self.pool = AttentionPool(enc_dim, pool_dim)
            pooled_dim = self.pool.out_dim
        elif pooling == "none":
            # No per-atom branch at all (the pf G-only arms): the trunk sees the
            # standardized descriptor block alone. Only meaningful with "concat" —
            # struct_only/resid have nothing to compute without a pooled input.
            if head_arch != "concat":
                raise ValueError('pooling "none" requires head_arch "concat"')
            pooled_dim = 0
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
    def fit_pool(self, x, seg, n):
        """Fit a learned pool's FROZEN projections (deepsets phi 'diag'/'pca') on
        the training per-atom rows; no-op for every other pooling."""
        pool = getattr(self, "pool", None)
        if pool is not None and getattr(pool, "needs_fit", False):
            pool.fit(x, seg, n)
        return self

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
        x is the per-atom (A,D) embedding with segment ids seg into n structures.
        "none": no per-atom branch — x is ignored entirely (may be None)."""
        if self.pooling == "none":
            return None
        if self.pooling == "meanmax":
            return self.pca(x)
        return self.pool(x, seg, n)

    def features(self, x, phys, seg=None, n=None) -> torch.Tensor:
        pooled = self._pooled(x, seg, n)
        if self.head_arch == "concat":
            ph = self.phys_std(phys)
            pooled = ph if pooled is None else torch.cat([pooled, ph], dim=1)
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
