#!/usr/bin/env python3
"""A/B the TF32-matmul speedup against exact fp32 on the CONSERVATIVE-FORCE path.

TF32 (`torch.set_float32_matmul_precision("high")`) is a free ~1.3-2x on Ampere+
matmuls, enabled by default in the trainer (see train._setup_matmul_precision). It
keeps full fp32 RANGE and only rounds matmul inputs to a 10-bit mantissa — far gentler
than the bf16 autocast we avoid for forces — but it is still reduced precision, so
this confirms it doesn't materially move the autograd forces/stress before trusting it
on a long run. We run ONE fixed batch through the multitask model twice — once exact
("highest"), once TF32 ("high") — and report the force/stress disagreement.

    python scripts/verify_tf32_forces.py     # exit 0 = pass / skipped, 1 = TF32 drift too large

TF32 is CUDA-matmul-only: on CPU there is nothing to compare, so this SKIPS (exit 0)
with a note to run it on the cluster GPU. Thresholds are loose — we only care that
TF32 isn't qualitatively wrong (the fp64 finite-difference correctness gate is
verify_autograd_forces.py); a few-e-3 relative force shift is expected and fine.
"""
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("models/common", "models/GPSTransformer"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import torch  # noqa: E402

# Reuse the exact synthetic multitask batch builder from the trainer smoke so the two
# gates can't drift in how they construct a graph/pack.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "verify_multitask_train", os.path.join(ROOT, "scripts", "verify_multitask_train.py"))
_vmt = importlib.util.module_from_spec(_spec)

REL_FORCE_TOL = 5e-2     # max relative force shift TF32 may introduce (loose: training signal)
REL_STRESS_TOL = 5e-2


def _build_model_and_batch():
    """A small multitask model + one collated batch, on CUDA. Mirrors
    verify_multitask_train.main()'s setup (synthetic 4-atom graphs, all 6 tasks)."""
    import json
    import tempfile
    import pandas as pd
    from torch.utils.data import DataLoader
    from data import load_cif_dataset, collate_pool_multitask, compute_feature_stats, DOS_N_ENERGY
    from pack import pack_dataset
    from model import GPSCrystalNet
    from train import _build_cart_strain, _to_input_var

    _spec.loader.exec_module(_vmt)        # gives us _vmt._graph / TASKS
    tmp = tempfile.mkdtemp(prefix="tf32_ab_")
    gd = os.path.join(tmp, "g"); os.makedirs(gd)
    ids, paths = [], []
    for i in range(16):
        p = os.path.join(gd, f"{i}.json")
        json.dump(_vmt._graph(i), open(p, "w"))
        ids.append(str(i)); paths.append(p)
    idx = pd.DataFrame({"id": ids, "graph_path": paths, "label": [1] * 16,
                        "formation_energy_per_atom": [-1.0 - 0.05 * i for i in range(16)],
                        "bandgap": [1.0 + 0.1 * i for i in range(16)]})
    ip = os.path.join(tmp, "idx.pickle"); idx.to_pickle(ip)
    pk = os.path.join(tmp, "pack"); pack_dataset(ip, pk, n_workers=1)
    ds = load_cif_dataset(pk, target_column="formation_energy_per_atom",
                          use_poly_edges=True, build_angle_bias=True, multitask=True)
    sa, sn, _, sp, _, _ = ds[0][0][:6]
    fs = compute_feature_stats(ds, list(range(len(ds))), max_graphs=16)
    model = GPSCrystalNet(sa.shape[-1], sn.shape[-1], poly_fea_len=sp.shape[-1],
                          atom_fea_len=64, n_conv=3, h_fea_len=64, n_h=2, n_heads=4,
                          use_poly_edges=True, gps_global=False, use_angle_bias=True,
                          tasks=_vmt.TASKS, differentiable_geometry=True, n_energy=DOS_N_ENERGY)
    model.set_feature_stats(fs["node"], fs["edge"], fs["poly"])
    model = model.cuda().train()
    loader = DataLoader(ds, batch_size=16, collate_fn=collate_pool_multitask)
    iv = _to_input_var(next(iter(loader))[0], cuda=True)
    return model, iv, _build_cart_strain


def _forces_stress(model, iv, build_cart_strain):
    cart, strain = build_cart_strain(iv)
    out = model(*iv, cart=cart, strain=strain)
    return out["forces"].detach().float(), out["stress"].detach().float()


def main():
    warnings.simplefilter("ignore")
    if not torch.cuda.is_available():
        print("verify_tf32_forces: SKIP (no CUDA — TF32 is a GPU-matmul feature; "
              "run this on the cluster a100 to A/B it).")
        return 0

    torch.manual_seed(0)
    model, iv, build_cart_strain = _build_model_and_batch()

    torch.set_float32_matmul_precision("highest")     # exact fp32 reference
    f_ref, s_ref = _forces_stress(model, iv, build_cart_strain)
    torch.set_float32_matmul_precision("high")         # TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    f_tf, s_tf = _forces_stress(model, iv, build_cart_strain)

    def _rel(a, b):
        denom = a.abs().mean().clamp_min(1e-8)
        return float((a - b).abs().max() / denom), float((a - b).abs().mean() / denom)

    f_max, f_mean = _rel(f_ref, f_tf)
    s_max, s_mean = _rel(s_ref, s_tf)
    ok = f_max < REL_FORCE_TOL and s_max < REL_STRESS_TOL
    print(f"  forces  rel-diff TF32 vs fp32: max={f_max:.2e} mean={f_mean:.2e} (tol {REL_FORCE_TOL})")
    print(f"  stress  rel-diff TF32 vs fp32: max={s_max:.2e} mean={s_mean:.2e} (tol {REL_STRESS_TOL})")
    print("verify_tf32_forces: " + ("PASS — TF32 safe for the force path" if ok
                                     else "FAIL — TF32 moves forces too much; set \"tf32\": false"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
