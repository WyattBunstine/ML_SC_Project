"""Physical-descriptor bypass for the T_c head: pooled crystal-level invariants.

One fixed-length vector per crystal, derived entirely from the stored cgv4
graph JSON (no rebuild, no positions). These are *inputs, not weights*: under
the SuperCon fresh-parameter budget they are free capacity, and in the
frozen-MACE configuration they are the only path the project's hand-crafted
chemistry reaches the head.

v1 layout (41 dims, names in DESCRIPTOR_NAMES):
  - mean+std over atoms of the 12 stored node features          (24)
  - log1p(n_atoms), bonding degree/atom, poly degree/atom        (3)
  - bond_length, bond_length_over_sum_radii, delta_chi_pauling
    mean+std over edges                                          (6)
  - poly: corner/edge/face share fractions, bridge-angle
    mean/std-of-means, direct-distance mean/std, path_type mean  (8)

NaNs in stored features (e.g. unresolved oxidation states) are zeroed, matching
the loader convention. Crystals with no poly edges get zeros in the poly block.
"""

import json
import os

import numpy as np

NODE_KEYS = [
    "Z", "oxidation_state", "ion_role", "chi_pauling", "chi_allen",
    "ecn_value", "shannon_radius", "cn_core",
    "hist_corner", "hist_edge", "hist_face", "hist_other",
]
EDGE_KEYS = ["bond_length", "bond_length_over_sum_radii", "delta_chi_pauling"]

DESCRIPTOR_NAMES = (
    [f"node_{k}_{s}" for k in NODE_KEYS for s in ("mean", "std")]
    + ["log1p_n_atoms", "bond_degree_per_atom", "poly_degree_per_atom"]
    + [f"edge_{k}_{s}" for k in EDGE_KEYS for s in ("mean", "std")]
    + ["poly_frac_corner", "poly_frac_edge", "poly_frac_face",
       "poly_mean_angle_mean", "poly_mean_angle_std",
       "poly_direct_distance_mean", "poly_direct_distance_std",
       "poly_path_type_mean"]
)
DESCRIPTOR_DIM = len(DESCRIPTOR_NAMES)


def _mean_std(values):
    arr = np.nan_to_num(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std())


def crystal_descriptors(graph) -> np.ndarray:
    """Fixed-order descriptor vector for one parsed graph JSON."""
    nodes, edges = graph["nodes"], graph.get("edges") or []
    poly = graph.get("poly_edges") or []
    n_atoms = max(len(nodes), 1)

    out = []
    for key in NODE_KEYS:
        out.extend(_mean_std([n.get(key, 0.0) for n in nodes]))

    out.append(float(np.log1p(len(nodes))))
    # Stored edges are canonical (one row per undirected edge) -> degree = 2E/N.
    out.append(2.0 * len(edges) / n_atoms)
    out.append(2.0 * len(poly) / n_atoms)

    for key in EDGE_KEYS:
        out.extend(_mean_std([e.get(key, 0.0) for e in edges]))

    if poly:
        shared = np.asarray([p.get("shared_count", 0) for p in poly])
        for count in (1, 2, 3):
            out.append(float((shared == count).mean()))
        out.extend(_mean_std([p.get("mean_angle_deg", 0.0) for p in poly]))
        out.extend(_mean_std([p.get("direct_distance", 0.0) for p in poly]))
        out.append(_mean_std([p.get("path_type", 0.0) for p in poly])[0])
    else:
        out.extend([0.0] * 8)

    vec = np.asarray(out, dtype=np.float32)
    assert vec.shape == (DESCRIPTOR_DIM,), vec.shape
    return vec


def build_descriptor_table(index_path, out_path,
                           prefix_map="database/MP:database/datafiles/MP",
                           progress_every=5000):
    """Compute descriptors for every index row -> pickle of {id: vector}."""
    import pandas as pd

    from CNN.head.embed_mace import remap_graph_path

    df = pd.read_pickle(index_path) if index_path.endswith(".pickle") else pd.read_csv(index_path)
    table, failed = {}, []
    for i, row in enumerate(df.itertuples()):
        try:
            with open(remap_graph_path(row.graph_path, prefix_map)) as f:
                table[row.id] = crystal_descriptors(json.load(f))
        except Exception as exc:
            failed.append((row.id, repr(exc)))
        if (i + 1) % progress_every == 0:
            print(f"  descriptors {i + 1}/{len(df)} ({len(failed)} failed)", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pd.to_pickle({"names": DESCRIPTOR_NAMES, "table": table, "failed": failed}, out_path)
    print(f"descriptors: {len(table)} ok, {len(failed)} failed -> {out_path}")
    return table
