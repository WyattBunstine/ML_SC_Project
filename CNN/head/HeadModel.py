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
"""

import torch
import torch.nn as nn


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
                 hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.pca = PCAWhiten(enc_dim, pca_k)
        self.phys_std = Standardizer(phys_dim)
        in_dim = pca_k + phys_dim
        self.norm = nn.LayerNorm(in_dim)
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.Softplus(),
                                   nn.Dropout(dropout))
        self.class_head = nn.Linear(hidden, 2)
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

    def features(self, enc: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        return self.trunk(self.norm(torch.cat([self.pca(enc), self.phys_std(phys)], dim=1)))

    def forward(self, enc: torch.Tensor, phys: torch.Tensor):
        h = self.features(enc, phys)
        return self.class_head(h), self.tc_head(h).squeeze(-1)

    def n_fresh_params(self) -> int:
        """Trainable parameters (buffers excluded) — the budgeted quantity."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
