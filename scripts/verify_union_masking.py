#!/usr/bin/env python3
"""Verify the masked-UNION semantics for multitask pretraining (ConcatMTDataset)
and the transfer-export contract that depend on them — the #13 review fixes.

Covers the failure modes the basic union smoke does NOT:
  - F3  A union member that LACKS the configured scalar target (the DOS pack has no
        formation_energy_per_atom) must LOAD (not raise), KEEP all its rows (not be
        emptied by the NaN-target drop), and have its energy NaN-masked while its own
        target (dos) still trains. The masked-union rule is "absent target -> masked,
        not dropped"; build_data_rows(keep_all) + _select_target_key(required=False).
  - F1  embed-gps must build the transfer dataset with target_column=None (the
        pretraining target is irrelevant + absent from the transfer index) so an index
        with tc/value but no formation energy resolves its own target instead of crashing.
  - from_args  GPSCrystalNet.from_args rebuilds the encoder spec (the single source both
        gps_main and embed_gps construct through, so the transfer encoder can't drift).
  - single-task keep_all=False stays bit-identical (no row-drop regression).

    python scripts/verify_union_masking.py     # exit 0 = pass, 1 = fail
"""
import json
import os
import shutil
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("models/common", "models/GPSTransformer", "models/head"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from data import load_cif_dataset, ConcatMTDataset, DOS_N_ENERGY  # noqa: E402
from pack import pack_dataset  # noqa: E402
from model import GPSCrystalNet  # noqa: E402
from train import compute_target_stats  # noqa: E402

warnings.simplefilter("ignore")

_NODE = {"Z": 11, "oxidation_state": 1.0, "ion_role": 1, "chi_pauling": 0.93,
         "chi_allen": 0.87, "ecn_value": 6.0, "shannon_radius": 1.0, "cn_core": 6,
         "hist_corner": 1, "hist_edge": 0, "hist_face": 0, "hist_other": 0,
         "ionization_energy": 5.0, "electron_affinity": 0.5}


def _edge(eid, s, t, bl):
    return {"id": eid, "source": s, "target": t, "bond_length": bl,
            "bond_length_over_sum_radii": bl / 2.0, "voronoi_weight_src": 0.5,
            "voronoi_weight_tgt": 0.5, "ecn_weight_src": 0.5, "ecn_weight_tgt": 0.5,
            "delta_chi_pauling": 0.1}


def _graph(seed, kind):
    n = 4
    rng = np.random.RandomState(seed)
    g = {"nodes": [dict(_NODE, Z=11 + i) for i in range(n)],
         "edges": [_edge(0, 0, 1, 2.0), _edge(1, 1, 2, 2.1), _edge(2, 2, 3, 2.2)],
         "adjacency": {"0": [[0, 1]], "1": [[0, 0], [1, 2]], "2": [[1, 1], [2, 3]], "3": [[2, 2]]},
         "poly_edges": [], "poly_adjacency": {str(i): [] for i in range(n)},
         "angle_triplets": [[1, 0, 1, 0.3], [2, 1, 2, -0.2]], "dihedrals": []}
    if kind == "phys":
        g["forces"] = (rng.rand(n, 3) - 0.5).tolist()
        g["magmom"] = rng.rand(n).tolist()
        g["stress"] = (rng.rand(3, 3) * 0.1).tolist()
    else:
        g["dos"] = np.abs(rng.randn(DOS_N_ENERGY)).tolist()
    return g


def _build_index(tmp, name, kind, nrows, extra_cols):
    gd = os.path.join(tmp, name)
    os.makedirs(gd)
    ids, paths = [], []
    for i in range(nrows):
        p = os.path.join(gd, f"{name}-{i}.json")
        json.dump(_graph(i, kind), open(p, "w"))
        ids.append(f"{name}-{i}")
        paths.append(p)
    d = {"id": ids, "graph_path": paths, "label": [1] * nrows,
         "mp_id": [f"mp-{name}-{i}" for i in range(nrows)]}
    d.update({c: f(nrows) for c, f in extra_cols.items()})
    ip = os.path.join(tmp, name + ".pickle")
    pd.DataFrame(d).to_pickle(ip)
    return ip


def main():
    tmp = tempfile.mkdtemp()
    try:
        eform = lambda n: [-1.0 - 0.05 * i for i in range(n)]
        # phys pack carries formation energy; the DOS pack deliberately does NOT.
        ip_phys = _build_index(tmp, "phys", "phys", 10, {"formation_energy_per_atom": eform})
        ip_dos = _build_index(tmp, "dos", "dos", 6, {})
        pk_phys = os.path.join(tmp, "phys_pk")
        pk_dos = os.path.join(tmp, "dos_pk")
        pack_dataset(ip_phys, pk_phys, n_workers=1)
        pack_dataset(ip_dos, pk_dos, n_workers=1)

        kw = dict(target_column="formation_energy_per_atom", use_poly_edges=True,
                  build_angle_bias=True, multitask=True)

        # F3: the eform-less DOS pack loads, keeps all rows, target_column -> None.
        ds_dos = load_cif_dataset(pk_dos, **kw)
        f3_load = (len(ds_dos) == 6 and ds_dos.target_column is None)
        m = ds_dos[0][2]
        f3_mask = (not bool(m["energy"])) and bool(m["dos"])

        # F3: union with the eform-less DOS pack collects target stats over BOTH packs.
        ds_phys = load_cif_dataset(pk_phys, **kw)
        u = ConcatMTDataset([ds_phys, ds_dos])
        ts = compute_target_stats(u, range(len(u)), max_samples=100)
        f3_union = (len(u) == 16 and {"dos", "energy", "forces"} <= set(ts))

        # F1: embed-gps transfer load — index with tc but no formation energy,
        #     target_column=None resolves the natural target without raising.
        tc = lambda n: [float(i % 5) for i in range(n)]
        ip_tc = _build_index(tmp, "sc", "phys", 8, {"tc": tc})
        ds_tc = load_cif_dataset(ip_tc, target_column=None, use_poly_edges=True,
                                 build_angle_bias=True, multitask=False)
        f1_ok = (len(ds_tc) == 8 and ds_tc.target_column == "tc")

        # single-task keep_all=False: a NaN scalar target STILL drops (no regression).
        nan_eform = lambda n: [(-1.0 if i % 2 == 0 else float("nan")) for i in range(n)]
        ip_drop = _build_index(tmp, "drop", "phys", 8,
                               {"formation_energy_per_atom": nan_eform})
        ds_drop = load_cif_dataset(ip_drop, target_column="formation_energy_per_atom",
                                   use_poly_edges=True, build_angle_bias=True, multitask=False)
        drop_ok = (len(ds_drop) == 4)        # 4 NaN rows dropped in single-task mode

        # from_args: rebuild the multitask encoder spec.
        fa = GPSCrystalNet.from_args(
            {"tasks": ["energy", "forces", "dos"], "atom_feat_len": 16, "n_conv": 2,
             "h_feat_len": 16, "gps_global": False, "set_transformer_heads": 2,
             "n_energy": DOS_N_ENERGY}, (14, 7, 7))
        fa_ok = (fa.tasks == {"energy", "forces", "dos"}
                 and fa.differentiable_geometry is True and fa.atom_fea_len == 16)

        ok = all([f3_load, f3_mask, f3_union, f1_ok, drop_ok, fa_ok])
        print(f"F3_load={f3_load} F3_mask={f3_mask} F3_union={f3_union} "
              f"F1_target_none={f1_ok} single_task_drop={drop_ok} from_args={fa_ok}")
        print("verify_union_masking: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(main())
