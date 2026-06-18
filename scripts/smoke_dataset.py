#!/usr/bin/env python3
"""End-to-end dataset/model contract smoke — catches "sample contract changed" bugs.

Builds a tiny synthetic cgv4 dataset (with positions), opens it through BOTH backends
(packed PackedCIFDataV4 + lazy CIFDataV4), and runs each model (MPNN, GPS) through the
full startup path that the cluster jobs use:

    dataset[i] -> feature-dim detection -> model build -> collate -> forward + backward

This is the class of bug that unit-testing the pieces misses: when the per-sample tuple
shape changes (e.g. adding frac_coords + lattice for the GPS distance bias), the model
forwards and collate may pass in isolation while the trainers' `dataset[0]` dim-detect
unpack and the lazy `_build_sample` quietly break — exactly what slipped past a piecewise
review and only surfaced on the cluster. Run before any data-layer change.

    python scripts/smoke_dataset.py        # exit 0 = pass, 1 = fail

Scope note: the angle-bias and size-grouped-sampler paths are covered by their own
checks; this guards the dataset -> collate -> model arity contract for both backends.
"""
import json
import os
import shutil
import sys
import tempfile
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("models/common", "models/MPNN", "models/GPSTransformer"):
    sys.path.insert(0, os.path.join(ROOT, _p))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from data import (load_cif_dataset, collate_pool, collate_pool_geom,  # noqa: E402
                  compute_feature_stats, RICH_NODE_FEA_LEN, DIHEDRAL_FEA_LEN)
from pack import pack_dataset  # noqa: E402
from train import _to_input_var  # noqa: E402
from model import GPSCrystalNet  # noqa: E402
from MPNNModel import CrystalMPNN  # noqa: E402

# A minimal valid compact node (the 14 features _node_to_fea reads).
_NODE = {"Z": 11, "oxidation_state": 1.0, "ion_role": 1, "chi_pauling": 0.93,
         "chi_allen": 0.87, "ecn_value": 6.0, "shannon_radius": 1.0, "cn_core": 6,
         "hist_corner": 1, "hist_edge": 0, "hist_face": 0, "hist_other": 0,
         "ionization_energy": 5.0, "electron_affinity": 0.5}


def _graph(n):
    """A tiny positioned graph (no bonds/poly/angles needed for the contract test)."""
    return {"nodes": [dict(_NODE, Z=11 + i) for i in range(n)], "edges": [],
            "adjacency": {str(i): [] for i in range(n)}, "poly_edges": [],
            "poly_adjacency": {str(i): [] for i in range(n)}, "angle_triplets": [],
            "dihedrals": [],
            "frac_coords": [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(n)],
            "lattice": [[4.0, 0.1, 0.0], [0.0, 4.0, 0.1], [0.1, 0.0, 4.0]]}


def _build_tmp_dataset(tmp):
    gd = os.path.join(tmp, "graphs")
    os.makedirs(gd)
    ids, paths = [], []
    for i, n in enumerate([3, 4, 5, 4, 6, 3]):       # varied sizes -> collate must pad
        p = os.path.join(gd, f"{i}.json")
        json.dump(_graph(n), open(p, "w"))
        ids.append(str(i))
        paths.append(p)
    idx = pd.DataFrame({"id": ids, "graph_path": paths, "label": [1] * len(ids),
                        "formation_energy_per_atom": [-1.0 - 0.1 * i for i in range(len(ids))]})
    index_pickle = os.path.join(tmp, "index.pickle")
    idx.to_pickle(index_pickle)
    pack_dir = os.path.join(tmp, "pack")
    pack_dataset(index_pickle, pack_dir, n_workers=1)
    return index_pickle, pack_dir


def _check_backend(name, dataset):
    # 1. feature-dim detection — the exact `dataset[0]` unpack the trainers do.
    sa, sn, _, sp, _, _ = dataset[0][0][:6]
    dims = (sa.shape[-1], sn.shape[-1], sp.shape[-1])
    samples = [dataset[i] for i in range(4)]

    # 2. MPNN: collate_pool (8-elem batch) -> CrystalMPNN forward/backward.
    mpnn_in, *_ = collate_pool(samples)
    _to_input_var(mpnn_in, cuda=False)                # exercise the arity-agnostic move
    mpnn = CrystalMPNN(dims[0], dims[1], poly_fea_len=dims[2],
                       atom_fea_len=16, n_conv=1, h_fea_len=16, n_h=1)
    mpnn.train()
    mo = mpnn(*mpnn_in)
    mo.pow(2).mean().backward()

    # 3. GPS: collate_pool_geom (10-elem batch) -> GPSCrystalNet(use_dist_bias) fwd/bwd.
    gps_in, *_ = collate_pool_geom(samples)
    _to_input_var(gps_in, cuda=False)
    gps = GPSCrystalNet(dims[0], dims[1], poly_fea_len=dims[2], atom_fea_len=16,
                        n_conv=2, h_fea_len=16, n_h=1, n_heads=2, gps_global=True,
                        gps_global_heads=2, gps_ffn_mult=2, use_angle_bias=False,
                        use_dist_bias=True)
    gps.train()
    go = gps(*gps_in)
    go.pow(2).mean().backward()

    ok = (dims == (14, 7, 7) and len(mpnn_in) == 8 and len(gps_in) == 10
          and tuple(mo.shape) == (4, 1) and tuple(go.shape) == (4, 1)
          and torch.isfinite(mo).all().item() and torch.isfinite(go).all().item())
    print(f"  {name:7} dims={dims} mpnn_batch={len(mpnn_in)} gps_batch={len(gps_in)} "
          f"mpnn_fwd={tuple(mo.shape)} gps_distbias_fwd={tuple(go.shape)} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def _check_rich(pack_dir, kw):
    # use_rich_node_features: atom_fea widens by RICH_NODE_FEA_LEN at assemble time
    # AND the (packed) feature-stats path must match it, else set_feature_stats blows
    # up on a dim mismatch (the exact bug this guards).
    ds = load_cif_dataset(pack_dir, use_rich_node_features=True, **kw)
    nd = ds[0][0][0].shape[-1]
    stats = compute_feature_stats(ds, list(range(len(ds))), max_graphs=10)
    dims = (nd, ds[0][0][1].shape[-1], ds[0][0][3].shape[-1])
    m = GPSCrystalNet(dims[0], dims[1], poly_fea_len=dims[2], atom_fea_len=16,
                      n_conv=1, h_fea_len=16, n_h=1, n_heads=2, gps_global=False)
    m.set_feature_stats(stats["node"], stats["edge"], stats["poly"])   # dim-match check
    m.train()
    out = m(*collate_pool_geom([ds[i] for i in range(4)])[0])
    base = ds[0][0][0].shape[-1] - RICH_NODE_FEA_LEN
    ok = (nd == base + RICH_NODE_FEA_LEN and len(stats["node"][0]) == nd
          and tuple(out.shape) == (4, 1) and torch.isfinite(out).all().item())
    print(f"  rich    atom_fea={nd} (+{RICH_NODE_FEA_LEN}) stats={len(stats['node'][0])} "
          f"fwd={tuple(out.shape)} {'PASS' if ok else 'FAIL'}")
    return ok


def _edge(eid, s, t, bl):
    return {"id": eid, "source": s, "target": t, "bond_length": bl,
            "bond_length_over_sum_radii": bl / 2.0, "voronoi_weight_src": 0.5,
            "voronoi_weight_tgt": 0.5, "ecn_weight_src": 0.5, "ecn_weight_tgt": 0.5,
            "delta_chi_pauling": 0.1}


def _chain_graph():
    # atoms 0-1-2-3 with bonds e0=(0,1) e1=(1,2) e2=(2,3); one torsion around the
    # CENTRAL bond e1 -> dih_node nonzero only on its endpoints (atoms 1, 2).
    return {"nodes": [dict(_NODE, Z=11 + i) for i in range(4)],
            "edges": [_edge(0, 0, 1, 2.0), _edge(1, 1, 2, 2.1), _edge(2, 2, 3, 2.2)],
            "adjacency": {"0": [[0, 1]], "1": [[0, 0], [1, 2]],
                          "2": [[1, 1], [2, 3]], "3": [[2, 2]]},
            "poly_edges": [], "poly_adjacency": {str(i): [] for i in range(4)},
            "angle_triplets": [], "dihedrals": [[1, 0, 2, -0.5]]}


def _check_dih(tmp):
    # use_dihedrals: a re-packed (has_dihedrals) pack widens atom_fea by
    # DIHEDRAL_FEA_LEN, the stats path must match, and both backends must agree.
    gd = os.path.join(tmp, "dih_graphs")
    os.makedirs(gd)
    ids, paths = [], []
    for i in range(4):
        p = os.path.join(gd, f"{i}.json")
        json.dump(_chain_graph(), open(p, "w"))
        ids.append(str(i))
        paths.append(p)
    idx = pd.DataFrame({"id": ids, "graph_path": paths, "label": [1] * 4,
                        "formation_energy_per_atom": [-1.0 - 0.1 * i for i in range(4)]})
    ip = os.path.join(tmp, "dih_index.pickle")
    idx.to_pickle(ip)
    pk = os.path.join(tmp, "dih_pack")
    from pack import pack_dataset as _pd
    _pd(ip, pk, n_workers=1)
    has_dih = json.load(open(os.path.join(pk, "pack_header.json"))).get("has_dihedrals")
    kw = dict(target_column="formation_energy_per_atom", use_poly_edges=True,
              use_bond_angles=False, build_angle_bias=False)
    ds = load_cif_dataset(pk, use_dihedrals=True, **kw)
    nd = ds[0][0][0].shape[-1]
    stats = compute_feature_stats(ds, list(range(len(ds))), max_graphs=10)
    nd_lazy = load_cif_dataset(ip, use_dihedrals=True, **kw)[0][0][0].shape[-1]
    dims = (nd, ds[0][0][1].shape[-1], ds[0][0][3].shape[-1])
    m = GPSCrystalNet(dims[0], dims[1], poly_fea_len=dims[2], atom_fea_len=16,
                      n_conv=1, h_fea_len=16, n_h=1, n_heads=2, gps_global=False)
    m.set_feature_stats(stats["node"], stats["edge"], stats["poly"])
    m.train()
    out = m(*collate_pool_geom([ds[i] for i in range(4)])[0])
    ok = (has_dih is True and nd == 14 + DIHEDRAL_FEA_LEN and len(stats["node"][0]) == nd
          and nd_lazy == nd and tuple(out.shape) == (4, 1) and torch.isfinite(out).all().item())
    print(f"  dih     atom_fea={nd} (+{DIHEDRAL_FEA_LEN}) stats={len(stats['node'][0])} "
          f"lazy={nd_lazy} has_dihedrals={has_dih} fwd={tuple(out.shape)} {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    warnings.simplefilter("ignore")
    tmp = tempfile.mkdtemp(prefix="smoke_dataset_")
    try:
        index_pickle, pack_dir = _build_tmp_dataset(tmp)
        kw = dict(target_column="formation_energy_per_atom", use_poly_edges=True,
                  use_bond_angles=False, build_angle_bias=False)
        ok_packed = _check_backend("packed", load_cif_dataset(pack_dir, **kw))
        ok_lazy = _check_backend("lazy", load_cif_dataset(index_pickle, **kw))
        ok_rich = _check_rich(pack_dir, kw)
        ok_dih = _check_dih(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if ok_packed and ok_lazy and ok_rich and ok_dih:
        print("smoke_dataset: PASS")
        return 0
    print("smoke_dataset: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
