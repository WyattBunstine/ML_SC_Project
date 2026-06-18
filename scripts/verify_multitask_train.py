#!/usr/bin/env python3
"""End-to-end smoke for the conservative-autograd multitask TRAINER (train.py).

Complements scripts/verify_autograd_forces.py (which proves the forces/stress are
correct): this drives the actual training path — MT pack -> multitask dataset ->
collate_pool_multitask -> GPSCrystalNet(tasks=...) -> compute_target_stats ->
_train_mt (the DOUBLE-backward step) -> _validate_mt — and checks it runs, the loss
descends, every active task's MAE is finite, and parameters receive gradients.

    python scripts/verify_multitask_train.py     # exit 0 = pass, 1 = fail
"""
import json
import math
import os
import shutil
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("models/common", "models/GPSTransformer"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from data import load_cif_dataset, collate_pool_multitask, compute_feature_stats  # noqa: E402
from pack import pack_dataset  # noqa: E402
from model import GPSCrystalNet  # noqa: E402
from train import (compute_target_stats, _train_mt, _validate_mt,  # noqa: E402
                   _DEFAULT_LOSS_WEIGHTS, _build_cart_strain, _to_input_var)

_NODE = {"Z": 11, "oxidation_state": 1.0, "ion_role": 1, "chi_pauling": 0.93,
         "chi_allen": 0.87, "ecn_value": 6.0, "shannon_radius": 1.0, "cn_core": 6,
         "hist_corner": 1, "hist_edge": 0, "hist_face": 0, "hist_other": 0,
         "ionization_energy": 5.0, "electron_affinity": 0.5}
TASKS = {"energy", "forces", "stress", "magmom", "bandgap"}


def _edge(eid, s, t, bl, ji=(0, 0, 0)):
    return {"id": eid, "source": s, "target": t, "bond_length": bl,
            "bond_length_over_sum_radii": bl / 2.0, "voronoi_weight_src": 0.5,
            "voronoi_weight_tgt": 0.5, "ecn_weight_src": 0.5, "ecn_weight_tgt": 0.5,
            "delta_chi_pauling": 0.1, "to_jimage": list(ji)}


def _graph(seed):
    rng = np.random.RandomState(seed)
    n = 4
    return {"nodes": [dict(_NODE, Z=11 + i + seed % 3) for i in range(n)],
            "edges": [_edge(0, 0, 1, 2.0 + 0.1 * (seed % 2)),
                      _edge(1, 1, 2, 2.1, ji=(1, 0, 0)), _edge(2, 2, 3, 2.2)],
            "adjacency": {"0": [[0, 1]], "1": [[0, 0], [1, 2]],
                          "2": [[1, 1], [2, 3]], "3": [[2, 2]]},
            "poly_edges": [], "poly_adjacency": {str(i): [] for i in range(n)},
            "angle_triplets": [[1, 0, 1, 0.3], [2, 1, 2, -0.2]], "dihedrals": [],
            "frac_coords": [[0.1 * i + 0.02 * seed, 0.2 * i, 0.3 * i] for i in range(n)],
            "lattice": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]],
            "forces": (rng.rand(n, 3) - 0.5).tolist(), "magmom": rng.rand(n).tolist(),
            "stress": (rng.rand(3, 3) * 0.1).tolist()}


def main():
    warnings.simplefilter("ignore")
    tmp = tempfile.mkdtemp(prefix="mt_train_")
    try:
        gd = os.path.join(tmp, "g")
        os.makedirs(gd)
        ids, paths = [], []
        for i in range(16):
            p = os.path.join(gd, f"{i}.json")
            json.dump(_graph(i), open(p, "w"))
            ids.append(str(i))
            paths.append(p)
        idx = pd.DataFrame({"id": ids, "graph_path": paths, "label": [1] * 16,
                            "formation_energy_per_atom": [-1.0 - 0.05 * i for i in range(16)],
                            "bandgap": [1.0 + 0.1 * i for i in range(16)]})
        ip = os.path.join(tmp, "idx.pickle")
        idx.to_pickle(ip)
        pk = os.path.join(tmp, "pack")
        pack_dataset(ip, pk, n_workers=1)

        ds = load_cif_dataset(pk, target_column="formation_energy_per_atom",
                              use_poly_edges=True, build_angle_bias=True, multitask=True)
        stats = compute_target_stats(ds, range(len(ds)), max_samples=100)
        sa, sn, _, sp, _, _ = ds[0][0][:6]
        fs = compute_feature_stats(ds, list(range(len(ds))), max_graphs=16)
        model = GPSCrystalNet(sa.shape[-1], sn.shape[-1], poly_fea_len=sp.shape[-1],
                              atom_fea_len=16, n_conv=2, h_fea_len=16, n_h=2, n_heads=2,
                              use_poly_edges=True, gps_global=False, use_angle_bias=True,
                              tasks=TASKS, differentiable_geometry=True, n_energy=8)
        model.set_feature_stats(fs["node"], fs["edge"], fs["poly"])
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_pool_multitask)
        args = {"cuda": False, "learning_rate": 0.01, "print_split": 1000,
                "grad_clip": 0.5, "warmup_epochs": 0}
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        losses = []
        for ep in range(12):
            tl, _tm, _, _ = _train_mt(loader, model, opt, ep, stats, _DEFAULT_LOSS_WEIGHTS, args)
            losses.append(tl)
        _vl, vm = _validate_mt(loader, model, stats, _DEFAULT_LOSS_WEIGHTS, args)
        gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)

        # CRITICAL: forces/stress must ACTUALLY contribute — a force-only (and stress-only)
        # objective must produce nonzero PARAMETER gradients through the double backward.
        # Without this, energy+bandgap alone drive the loss descent and a silently
        # zeroed/detached force head would still "pass" descends + finite + gnorm>0.
        model.train()
        iv = _to_input_var(next(iter(loader))[0], False)
        cart, strain = _build_cart_strain(iv)
        o = model(*iv, cart=cart, strain=strain)

        def _param_gnorm(objective):
            model.zero_grad(set_to_none=True)
            objective.backward(retain_graph=True)
            return sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)

        f_g = _param_gnorm(o["forces"].abs().sum())     # double backward through the force path
        s_g = _param_gnorm(o["stress"].abs().sum())
        contribute = f_g > 0 and s_g > 0

        descends = losses[-1] < losses[0] * 0.9
        finite = all(math.isfinite(v) for v in losses) and all(math.isfinite(v) for v in vm.values())
        tasks_ok = set(vm) == TASKS
        ok = descends and finite and tasks_ok and gnorm > 0 and contribute
        print(f"  target std: {{{', '.join(f'{k}:{v:.3f}' for k, v in stats.items())}}}")
        print(f"  train loss {losses[0]:.2f} -> {losses[-1]:.2f} (descends={descends})")
        print(f"  val MAE: {{{', '.join(f'{k}:{vm[k]:.4f}' for k in sorted(vm))}}}")
        print(f"  double-backward param-grad-norm={gnorm:.2e}  tasks={tasks_ok}")
        print(f"  force/stress contribute to params: F={f_g:.2e} S={s_g:.2e}  ({contribute})")
        print("verify_multitask_train: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
