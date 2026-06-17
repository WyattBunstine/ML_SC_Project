from __future__ import print_function, division

import json
import os
import random
import warnings
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from pymatgen.core.periodic_table import Element as PmgElement
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

torch.manual_seed(0)

# --- Feature layout constants ---
# Node features (14 total), in order:
#   Z, oxidation_state, ion_role, chi_pauling, chi_allen,
#   ecn_value, shannon_radius, cn_core,
#   hist_corner, hist_edge, hist_face, hist_other,  (← 4 sharing histogram values)
#   ionization_energy, electron_affinity  (← free-atom electronic props, by Z)
# The last two are pure per-element lookups (carried over from the original
# CGCNN atom_init vector); they depend only on Z so they're computed from the
# node's atomic number rather than read from the stored graph — no DB rebuild
# needed for graphs that predate these features.
NODE_FEA_LEN = 14

# Edge features (7 total), in order:
#   bond_length, bond_length_over_sum_radii,
#   voronoi_weight_center, voronoi_weight_nbr,
#   ecn_weight_center, ecn_weight_nbr,
#   delta_chi_pauling
# The src/tgt directional pairs are oriented CENTER-relative at neighbor-list
# build time (see _edge_to_fea): 'center' is the atom being aggregated onto.
# coord_sphere was dropped: it is core ⟺ max(ecn_center, ecn_nbr) ≥ threshold, so
# it is fully determined by the two ECoN columns the model now sees (|r|≈0.90 with
# ecn alone) — a redundant input. Still stored in the graph JSON, just not fed.
NBR_FEA_LEN = 7

# Index of ecn_weight_center inside the edge feature vector — the center atom's
# own ECoN weight for this bond, used as the physical aggregation weight.
ECN_WEIGHT_SRC_IDX = 4

# Polyhedral edge features (7 total), in order:
#   shared_count, mean_angle_deg, std_angle_deg,
#   mean_path_length, std_path_length, direct_distance, path_type
# These are second-neighbour connections through a shared bridging atom
# (corner/edge/face sharing) — a distinct edge type from the bonding edges.
POLY_FEA_LEN = 7

# Index of shared_count inside the poly feature vector. Plays the same role
# for poly-edge aggregation that ecn_weight_src plays for bonding edges:
# face-sharing (3) > edge-sharing (2) > corner-sharing (1) connections get
# proportionally more weight.
POLY_WEIGHT_IDX = 0

# --- Bond-angle (3-body) RBF basis -----------------------------------------
# Each stored triplet is a cos(theta) for a pair of bonding edges meeting at a
# center atom. We expand it in this Gaussian RBF over cos in [-1, 1] and sum the
# expansions onto the two participating edges (per center), giving each bonding
# edge an "angular environment" vector that is concatenated to its
# NBR_FEA_LEN base features. The basis lives here (not in the stored graph) so it can be retuned
# without rebuilding the graphs. cos(theta) is used (not theta) so the basis is
# linear in the dot product and dense where bond angles cluster (90/109.5/180).
ANGLE_RBF_CENTERS = np.linspace(-1.0, 1.0, 12).astype(np.float32)
ANGLE_RBF_WIDTH = float(ANGLE_RBF_CENTERS[1] - ANGLE_RBF_CENTERS[0])  # center spacing
ANGLE_FEA_LEN = int(len(ANGLE_RBF_CENTERS))


def _angle_rbf(cos_vals) -> np.ndarray:
    """Gaussian RBF expansion of cos(theta) values -> (len(cos_vals), ANGLE_FEA_LEN)."""
    c = np.asarray(cos_vals, dtype=np.float32).reshape(-1, 1)
    diff = c - ANGLE_RBF_CENTERS.reshape(1, -1)
    return np.exp(-(diff ** 2) / (2.0 * ANGLE_RBF_WIDTH ** 2)).astype(np.float32)


def _build_edge_angle_feats(graph) -> dict:
    """(edge_id, center_atom) -> MEAN RBF(cos) over the bond-angle triplets that
    edge participates in at that center. Empty dict if the graph has no triplets
    (older graphs / feature disabled), in which case edges get a zero angle vector.

    Mean (not sum) so the magnitude doesn't scale with coordination number — the
    angular *shape* is what matters, and coordination is already a node feature
    (cn_core / ecn_value). This keeps the feature bounded and the dynamic range
    tight across low- and high-coordination atoms.

    Compact triplet form: [center, edge_a, edge_b, cos_angle].
    """
    triplets = graph.get("angle_triplets") or []
    if not triplets:
        return {}
    rbf = _angle_rbf([t[3] for t in triplets])  # (T, K)
    sums: dict = {}
    counts: dict = {}
    for (center, ea, eb, _cos), r in zip(triplets, rbf):
        for eid in (ea, eb):
            key = (eid, center)
            if key in sums:
                sums[key] += r
                counts[key] += 1
            else:
                sums[key] = r.copy()
                counts[key] = 1
    return {key: sums[key] / counts[key] for key in sums}


@lru_cache(maxsize=128)
def _element_electronic_props(z: int) -> tuple:
    """(ionization_energy, electron_affinity) in eV for atomic number ``z``.

    Pure per-element lookup via pymatgen, cached by Z. Missing values (pymatgen
    returns None for some elements) are coerced to 0.0 so they don't poison the
    feature vector. Matches the source the original CGCNN atom_init used.
    """
    el = PmgElement.from_Z(int(z))
    ie = el.ionization_energy
    ea = el.electron_affinity
    return (float(ie) if ie is not None else 0.0,
            float(ea) if ea is not None else 0.0)


def _node_to_fea(node: dict) -> np.ndarray:
    # Prefer values stored on the graph (newer graphs carry them); fall back to
    # the Z-based lookup for graphs built before these features were added. Coerce
    # the stored values the same way the lookup does (None/missing -> 0.0, cast to
    # float) so a JSON null or stringified number can't slip a NaN/non-numeric
    # into the feature vector.
    if "ionization_energy" in node and "electron_affinity" in node:
        ie, ea = node["ionization_energy"], node["electron_affinity"]
        ionization_energy = float(ie) if ie is not None else 0.0
        electron_affinity = float(ea) if ea is not None else 0.0
    else:
        ionization_energy, electron_affinity = _element_electronic_props(node["Z"])
    # nan_to_num: some per-element properties are genuinely undefined for certain
    # elements — e.g. Pauling electronegativity (chi_pauling) is NaN for the noble
    # gases (He, Ne, Ar). Left raw, a single such atom poisons compute_feature_stats
    # (NaN mean/std) and then every normalized forward pass, giving NaN loss/MAE
    # from batch 0. Coerce undefined values to 0.0, matching how ionization_energy /
    # electron_affinity already handle their missing values above.
    return np.nan_to_num(np.array([
        node["Z"],
        node["oxidation_state"],
        node["ion_role"],
        node["chi_pauling"],
        node["chi_allen"],
        node["ecn_value"],
        node["shannon_radius"],
        node["cn_core"],
        node["hist_corner"],
        node["hist_edge"],
        node["hist_face"],
        node["hist_other"],
        ionization_energy,
        electron_affinity,
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _center_is_source(edge: dict, center: int, nbr: int = None) -> bool:
    """Is ``center`` the stored source endpoint of ``edge``?

    Used to orient the directional edge features to the center atom. Newer
    graphs carry explicit source/target keys. Legacy graphs don't, but the
    builder canonicalizes every edge to source <= target (_canonical_edge in
    crystal_graph_v4; verified empirically across thousands of edges), so when
    the neighbor id is known the source is simply the smaller endpoint — i.e.
    legacy graphs orient correctly too, NO rebuild required. Self-image edges
    (center == nbr) resolve to source, which is exact: both ends are the same
    atom. Only with neither key nor neighbor do we fall back to stored order.
    """
    src = edge.get("source")
    if src is not None:
        return int(src) == int(center)
    if nbr is not None:
        return int(center) <= int(nbr)
    return True


def _edge_to_fea(edge: dict, center_is_source: bool = True) -> np.ndarray:
    # The src/tgt columns (voronoi_weight, ecn_weight) are DIRECTIONAL: as stored,
    # 'src' is the edge's source endpoint. For message passing we want them
    # center-relative — column 'center' is the atom we're aggregating onto — so we
    # swap the pair when the center is actually the target. Index 4 (ecn_center)
    # is then the center atom's own ECoN weight, which is what the weighted
    # aggregation in MPNNConvLayer reads (ECN_WEIGHT_SRC_IDX). bond_length,
    # len/sumR and delta_chi (|Δχ|) are symmetric, so unaffected.
    voro_c, voro_n = edge["voronoi_weight_src"], edge["voronoi_weight_tgt"]
    ecn_c, ecn_n = edge["ecn_weight_src"], edge["ecn_weight_tgt"]
    if not center_is_source:
        voro_c, voro_n = voro_n, voro_c
        ecn_c, ecn_n = ecn_n, ecn_c
    # nan_to_num for the same reason as _node_to_fea: delta_chi_pauling is NaN on
    # any edge touching a noble-gas atom (undefined Pauling electronegativity).
    return np.nan_to_num(np.array([
        edge["bond_length"],
        edge["bond_length_over_sum_radii"],
        voro_c,
        voro_n,
        ecn_c,
        ecn_n,
        edge["delta_chi_pauling"],
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _poly_edge_to_fea(pe: dict) -> np.ndarray:
    # shared_count first so it sits at POLY_WEIGHT_IDX (== 0).
    # nan_to_num guards against undefined geometry stats (e.g. std_angle_deg from a
    # single-sample angle set) leaking NaN into the feature stats / forward pass.
    return np.nan_to_num(np.array([
        pe["shared_count"],
        pe["mean_angle_deg"],
        pe["std_angle_deg"],
        pe["mean_path_length"],
        pe["std_path_length"],
        pe["direct_distance"],
        pe["path_type"],
    ], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _select_target_key(index_df, target_column, index_path):
    """Pick which column supplies the regression target. A multi-target index
    (e.g. the MP energy dataset) carries several named target columns;
    `target_column` (from the config) selects one. With it unset we fall back to
    the legacy `value` column, then `tc`, so older single-target indexes keep
    working unchanged. The fallback only accepts a column that actually has data
    — a tc-less index has an all-empty `value`, so this raises and tells the user
    to set `target_column` instead of silently regressing on the wrong target.
    Shared by CIFDataV4 (index pickle) and PackedCIFDataV4 (pack meta).
    mp_id is the grouping key for material splits, NOT a target."""
    structural_cols = {"id", "value", "graph_path", "label", "mp_id"}
    named_targets = [c for c in index_df.columns
                     if c not in structural_cols and not index_df[c].isna().all()]
    if target_column is not None:
        if target_column not in index_df.columns:
            raise ValueError(
                f"target_column '{target_column}' not found in index "
                f"{index_path}; available columns: {list(index_df.columns)}")
        return target_column
    target_key = next(
        (c for c in ("value", "tc")
         if c in index_df.columns and not index_df[c].isna().all()),
        None)
    if target_key is None:
        raise ValueError(
            f"No usable default target ('value'/'tc' absent or all-empty) "
            f"in index {index_path}. Set 'target_column' in the config to "
            f"one of {named_targets}.")
    return target_key


def build_data_rows(index_df, target_key, third_full, random_seed):
    """Shared row construction for both dataset backends (CIFDataV4 and
    PackedCIFDataV4): drop target-NaN rows, attach label/mp_id, seed-shuffle.

    The two backends MUST stay bit-identical here — the same seed has to produce
    the same ordering (hence the same train/val/test splits) regardless of which
    backend serves the samples. ``third_full`` is the backend-specific element 2
    of each row tuple (graph paths for the lazy loader, meta row positions for
    the packed one), aligned with the FULL index_df.

    Returns (data, groups, labels, dropped):
      data   : [(id, target_value, third, label), ...] shuffled
      groups : parallel mp_id list, or None unless EVERY row has one
               (all-or-nothing: a partially-populated grouping key can't support
               a leakage-free material split)
      labels : [label, ...] aligned with data
    """
    valid = index_df[target_key].notna()
    dropped = int((~valid).sum())
    sub = index_df.loc[valid]
    third = [t for t, v in zip(third_full, valid.tolist()) if v]
    labels = (sub["label"].astype(int).tolist()
              if "label" in sub.columns else [1] * len(sub))
    if "mp_id" in index_df.columns:
        mp_ids = [None if pd.isna(g) else g for g in sub["mp_id"].tolist()]
    else:
        mp_ids = [None] * len(sub)
    rows = list(zip(sub["id"].tolist(), sub[target_key].tolist(), third,
                    labels, mp_ids))
    random.seed(random_seed)
    random.shuffle(rows)
    data = [r[:4] for r in rows]
    group_vals = [r[4] for r in rows]
    groups = group_vals if group_vals and all(g is not None for g in group_vals) else None
    return data, groups, [rec[3] for rec in data], dropped


def accumulate_slot_rbf(vslots, vcos, n_slots):
    """Per-slot sequential float32 accumulation of RBF(cos) contributions.

    Returns (acc (n_slots, ANGLE_FEA_LEN), cnt (n_slots,)); divide acc by cnt
    where cnt > 0 for the mean. Accumulation runs in the given (graph) order in
    float32, matching _build_edge_angle_feats bitwise — shared by the sample
    assembly and the packed stats path so the angle featurization can't drift.
    """
    acc = np.zeros((n_slots, ANGLE_FEA_LEN), dtype=np.float32)
    cnt = np.zeros(n_slots, dtype=np.int64)
    if len(vcos):
        rbf = _angle_rbf(vcos)
        for s, row in zip(vslots, rbf):
            if s < n_slots:
                acc[s] += row
                cnt[s] += 1
    return acc, cnt


def rows_meanstd(rows, dim):
    """nan-safe per-column (mean, std) over feature rows; std floor 1.0 for
    zero-variance columns. Shared by the lazy and packed stats paths."""
    if not len(rows):
        return (np.zeros(dim, np.float32), np.ones(dim, np.float32))
    arr = np.stack(rows).astype(np.float64) if isinstance(rows, list) else np.asarray(rows, dtype=np.float64)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std[std < 1e-6] = 1.0
    return (mean.astype(np.float32), std.astype(np.float32))


def compute_feature_stats(dataset, indices, max_graphs=4000, seed=123):
    """Per-feature mean/std for node, bonding-edge, and poly-edge features.

    Computed over (a random sample of) the given *training* indices only — no
    val/test leakage. Stats come from the real graph entries (padding is never
    included). Zero-variance columns get std = 1.0 so normalization just centers
    them instead of dividing by ~0.

    Returns {"node": (mean, std), "edge": (mean, std), "poly": (mean, std)} as
    float32 arrays of length NODE_FEA_LEN / NBR_FEA_LEN / POLY_FEA_LEN.
    """
    # Packed datasets compute stats straight from their columnar arrays.
    if hasattr(dataset, "feature_stats"):
        return dataset.feature_stats(indices, max_graphs=max_graphs, seed=seed)

    idx = list(indices)
    if max_graphs and len(idx) > max_graphs:
        idx = random.Random(seed).sample(idx, max_graphs)

    # When bond angles are on, edge rows are the (NBR_FEA_LEN + ANGLE_FEA_LEN) vectors
    # the model actually consumes, so the normalizer matches the widened edges.
    use_ang = getattr(dataset, "use_bond_angles", False)
    edge_dim = NBR_FEA_LEN + (ANGLE_FEA_LEN if use_ang else 0)

    node_rows, edge_rows, poly_rows = [], [], []
    zero_ang = np.zeros(ANGLE_FEA_LEN, dtype=np.float32)
    for i in idx:
        graph = dataset._read_graph(dataset.data[i][2])
        node_rows.extend(_node_to_fea(n) for n in graph["nodes"])
        # Iterate every directed (edge, center) use exactly as __getitem__ does, so
        # the stats reflect the CENTER-relative orientation the model consumes
        # (each undirected edge contributes once per endpoint). With angles on,
        # triplet-less edges keep a zero angular vector — those structural zeros
        # belong in the stats because the model sees them too.
        acc = _build_edge_angle_feats(graph) if use_ang else {}
        edges_by_id = {e["id"]: e for e in graph["edges"]}
        for center_s, lst in graph.get("adjacency", {}).items():
            center = int(center_s)
            for eid, nbr in lst:
                e = edges_by_id.get(eid)
                if e is None:
                    continue
                fea = _edge_to_fea(e, _center_is_source(e, center, nbr))
                if use_ang:
                    fea = np.concatenate([fea, acc.get((eid, center), zero_ang)])
                edge_rows.append(fea)
        poly_rows.extend(_poly_edge_to_fea(pe) for pe in graph.get("poly_edges", []))

    # nan-safe mean/std via the shared helper (see rows_meanstd).
    return {
        "node": rows_meanstd(node_rows, NODE_FEA_LEN),
        "edge": rows_meanstd(edge_rows, edge_dim),
        "poly": rows_meanstd(poly_rows, POLY_FEA_LEN),
    }


def _extract_ragged(graph):
    """Extract a graph's UNPADDED, center-oriented neighbor data as flat arrays.

    The single source of truth shared by the lazy loader (`_build_sample`) and
    the packed-dataset writer (MPNNPack): both produce samples via
    `_assemble_sample` over this representation, so packed tensors are identical
    to lazily-built ones by construction.

    Per atom (center), neighbor lists are FULL (no max_num_nbr truncation —
    truncation/padding is an assemble-time decision, which is what lets one pack
    serve any max_num_nbr / max_num_poly_nbr / angle-flag combination), sorted
    exactly as _build_sample always sorted them (bond_length / direct_distance,
    stable). Bond features are center-oriented (_edge_to_fea + _center_is_source).
    Angle triplets are stored per center as (slot_a, slot_b, cos) where slots
    index the center's full sorted bond list.

    Returns a dict of numpy arrays:
      n_atoms   : int
      atom_fea  : (N, NODE_FEA_LEN) f32
      bond_cnt  : (N,) i32      per-center real bond count
      bond_nbr  : (E,) i32      neighbor LOCAL atom ids, grouped by center
      bond_fea  : (E, NBR_FEA_LEN) f32
      poly_cnt  : (N,) i32 ; poly_nbr : (P,) i32 ; poly_fea : (P, POLY_FEA_LEN) f32
      ang_cnt   : (N,) i32 ; ang_slots: (T, 2) i16 ; ang_cos : (T,) f32
    """
    nodes = graph["nodes"]
    n_atoms = len(nodes)
    atom_fea = (np.stack([_node_to_fea(n) for n in nodes], axis=0)
                if nodes else np.zeros((0, NODE_FEA_LEN), dtype=np.float32))

    edges_by_id = {e["id"]: e for e in graph["edges"]}
    adjacency = graph.get("adjacency", {})

    bond_cnt = np.zeros(n_atoms, dtype=np.int32)
    bond_nbr, bond_fea, slot_eids_all = [], [], []
    for atom_i in range(n_atoms):
        neighbors = adjacency.get(str(atom_i), [])

        def _bond_len(pair):
            e = edges_by_id.get(pair[0])
            return e["bond_length"] if e is not None else 1e9

        eids = []
        for eid, nbr_id in sorted(neighbors, key=_bond_len):
            edge = edges_by_id.get(eid)
            if edge is None:
                continue
            bond_fea.append(_edge_to_fea(edge, _center_is_source(edge, atom_i, nbr_id)))
            bond_nbr.append(nbr_id)
            eids.append(eid)
        bond_cnt[atom_i] = len(eids)
        slot_eids_all.append(eids)

    poly_cnt = np.zeros(n_atoms, dtype=np.int32)
    poly_nbr, poly_fea = [], []
    poly_by_id = {pe["id"]: pe for pe in graph.get("poly_edges", [])}
    poly_adjacency = graph.get("poly_adjacency", {})
    for atom_i in range(n_atoms):
        neighbors = poly_adjacency.get(str(atom_i), [])

        def _poly_dist(pair):
            pe = poly_by_id.get(pair[0])
            return pe["direct_distance"] if pe is not None else 1e9

        k = 0
        for pid, nbr_id in sorted(neighbors, key=_poly_dist):
            pe = poly_by_id.get(pid)
            if pe is None:
                continue
            poly_fea.append(_poly_edge_to_fea(pe))
            poly_nbr.append(nbr_id)
            k += 1
        poly_cnt[atom_i] = k

    # Angle triplets re-keyed from edge ids to the center's slot positions in its
    # full sorted bond list (graph order preserved — assembly relies on it for
    # bitwise-identical accumulation/fill order). Two streams because the two
    # consumers historically resolved duplicate slots differently (a self-image
    # edge occupies TWO slots of its center's list with the same edge id):
    #   pair stream (ang_slots/ang_cos)  — for the (M, M) bias matrix; slots via
    #       last-wins dict lookup, matching the original slot_of behavior.
    #   per-slot stream (ang_v*)         — for the mean-RBF edge vectors; the
    #       original keyed its accumulator by EDGE id, so every slot holding
    #       that edge gets the contribution — expand across duplicates.
    trip_by_center = {}
    for t in graph.get("angle_triplets", []):
        trip_by_center.setdefault(int(t[0]), []).append(
            (int(t[1]), int(t[2]), float(t[3])))
    ang_cnt = np.zeros(n_atoms, dtype=np.int32)
    ang_slots, ang_cos = [], []
    ang_vcnt = np.zeros(n_atoms, dtype=np.int32)
    ang_vslot, ang_vcos = [], []
    for atom_i in range(n_atoms):
        eids = slot_eids_all[atom_i]
        slot_of = {e: s for s, e in enumerate(eids)}          # last-wins
        slots_multi = {}
        for s, e in enumerate(eids):
            slots_multi.setdefault(e, []).append(s)
        k = kv = 0
        for ea, eb, cosv in trip_by_center.get(atom_i, ()):
            sa, sb = slot_of.get(ea), slot_of.get(eb)
            if sa is None or sb is None:
                continue
            ang_slots.append((sa, sb))
            ang_cos.append(cosv)
            k += 1
            for s in slots_multi[ea]:
                ang_vslot.append(s); ang_vcos.append(cosv); kv += 1
            for s in slots_multi[eb]:
                ang_vslot.append(s); ang_vcos.append(cosv); kv += 1
        ang_cnt[atom_i] = k
        ang_vcnt[atom_i] = kv

    def _arr(lst, dtype, width=None):
        if lst:
            return np.asarray(lst, dtype=dtype)
        return np.zeros((0,) if width is None else (0, width), dtype=dtype)

    # Geometry for the long-range distance bias (optional): atom-aligned fractional
    # coords (N,3) + the lattice (3,3). Absent in legacy graphs -> zeros (the model's
    # distance bias is gated off / warns when positions are all-zero). frac_coords
    # reuses the atom alignment, so the packer can index it with atom_start/n_atoms.
    fc = graph.get("frac_coords")
    frac_coords = (np.asarray(fc, dtype=np.float32).reshape(n_atoms, 3)
                   if fc is not None and len(fc) == n_atoms
                   else np.zeros((n_atoms, 3), dtype=np.float32))
    lat = graph.get("lattice")
    lattice = (np.asarray(lat, dtype=np.float32).reshape(3, 3)
               if lat is not None else np.zeros((3, 3), dtype=np.float32))

    return {
        "n_atoms": n_atoms,
        "atom_fea": atom_fea.astype(np.float32),
        "frac_coords": frac_coords,
        "lattice": lattice,
        "bond_cnt": bond_cnt,
        "bond_nbr": _arr(bond_nbr, np.int32),
        "bond_fea": _arr(bond_fea, np.float32, NBR_FEA_LEN),
        "poly_cnt": poly_cnt,
        "poly_nbr": _arr(poly_nbr, np.int32),
        "poly_fea": _arr(poly_fea, np.float32, POLY_FEA_LEN),
        "ang_cnt": ang_cnt,
        "ang_slots": _arr(ang_slots, np.int16, 2).reshape(-1, 2),
        "ang_cos": _arr(ang_cos, np.float32),
        "ang_vcnt": ang_vcnt,
        "ang_vslot": _arr(ang_vslot, np.int16),
        "ang_vcos": _arr(ang_vcos, np.float32),
    }


def _assemble_sample(r, max_num_nbr, max_num_poly_nbr,
                     use_poly_edges, use_bond_angles, build_angle_bias):
    """Truncate/pad a ragged extraction into the model's padded sample tensors.

    Replicates the historical _build_sample behavior exactly: closest-first
    truncation, zero-feature padding, self-loop index padding, per-(edge,center)
    mean-RBF angle vectors (use_bond_angles), and the (M, M) cos matrix with
    sentinel 2.0 (build_angle_bias). Returns (atom_fea, nbr_fea, nbr_fea_idx,
    poly_fea, poly_fea_idx, nbr_angle) as CPU torch tensors.
    """
    n_atoms = int(r["n_atoms"])
    M = max_num_nbr
    edge_width = NBR_FEA_LEN + (ANGLE_FEA_LEN if use_bond_angles else 0)

    nbr_fea = np.zeros((n_atoms, M, edge_width), dtype=np.float32)
    nbr_idx = np.repeat(np.arange(n_atoms, dtype=np.int64)[:, None], M, axis=1)

    b0 = v0 = 0
    for i in range(n_atoms):
        c = int(r["bond_cnt"][i])
        k = min(c, M)
        nbr_fea[i, :k, :NBR_FEA_LEN] = r["bond_fea"][b0:b0 + k]
        nbr_idx[i, :k] = r["bond_nbr"][b0:b0 + k]

        v = int(r["ang_vcnt"][i])
        if use_bond_angles and v:
            # Mean RBF(cos) per kept slot from the per-slot contribution stream
            # (see accumulate_slot_rbf — bitwise-matches _build_edge_angle_feats,
            # incl. duplicate-slot self-image edges).
            acc, cnt = accumulate_slot_rbf(r["ang_vslot"][v0:v0 + v],
                                           r["ang_vcos"][v0:v0 + v], k)
            nz = cnt > 0
            nbr_fea[i, :k][nz, NBR_FEA_LEN:] = acc[nz] / cnt[nz, None]

        b0 += c
        v0 += v

    if build_angle_bias:
        # Vectorized (M, M) bias fill across all atoms at once: a per-triplet
        # Python loop was ~90% of packed read time. Interleaving the (sa,sb) and
        # (sb,sa) writes per triplet before one fancy assignment reproduces the
        # loop's write ORDER exactly (numpy assigns duplicate indices in order),
        # so last-write-wins semantics — hence outputs — are bitwise unchanged.
        amat = np.full((n_atoms, M, M), 2.0, dtype=np.float32)
        n_trip = int(r["ang_cnt"].sum()) if n_atoms else 0
        if n_trip:
            atoms = np.repeat(np.arange(n_atoms), np.asarray(r["ang_cnt"]))
            slots = np.asarray(r["ang_slots"], dtype=np.int64)
            cos = np.asarray(r["ang_cos"], dtype=np.float32)
            sa, sb = slots[:, 0], slots[:, 1]
            keep = (sa < M) & (sb < M)
            if keep.any():
                a_k, sa_k, sb_k, c_k = atoms[keep], sa[keep], sb[keep], cos[keep]
                n = len(a_k)
                rows = np.empty(2 * n, dtype=np.int64)
                cols = np.empty(2 * n, dtype=np.int64)
                rows[0::2], rows[1::2] = sa_k, sb_k
                cols[0::2], cols[1::2] = sb_k, sa_k
                amat[np.repeat(a_k, 2), rows, cols] = np.repeat(c_k, 2)
        nbr_angle_arr = amat

    # copy=True: r["atom_fea"] may be a read-only memmap view (packed dataset) —
    # wrapping it directly would emit a non-writable warning and alias pack data.
    atom_fea = torch.from_numpy(np.array(r["atom_fea"], dtype=np.float32, copy=True))
    nbr_fea_t = torch.from_numpy(nbr_fea)
    nbr_fea_idx = torch.from_numpy(nbr_idx)
    if build_angle_bias:
        nbr_angle = torch.from_numpy(nbr_angle_arr)
    else:
        nbr_angle = torch.zeros((n_atoms, 0, 0), dtype=torch.float32)

    if use_poly_edges:
        Mp = max_num_poly_nbr
        poly_fea = np.zeros((n_atoms, Mp, POLY_FEA_LEN), dtype=np.float32)
        poly_idx = np.repeat(np.arange(n_atoms, dtype=np.int64)[:, None], Mp, axis=1)
        p0 = 0
        for i in range(n_atoms):
            c = int(r["poly_cnt"][i])
            k = min(c, Mp)
            poly_fea[i, :k] = r["poly_fea"][p0:p0 + k]
            poly_idx[i, :k] = r["poly_nbr"][p0:p0 + k]
            p0 += c
        poly_fea_t = torch.from_numpy(poly_fea)
        poly_fea_idx = torch.from_numpy(poly_idx)
    else:
        poly_fea_t = torch.zeros((n_atoms, 0, POLY_FEA_LEN), dtype=torch.float32)
        poly_fea_idx = torch.zeros((n_atoms, 0), dtype=torch.long)

    # Geometry for the long-range distance bias (zeros when the source has none).
    # Appended to the sample tuple; collate_pool ignores them, collate_pool_geom
    # (GPS) batches them. copy=True: r["frac_coords"] may be a read-only memmap view.
    frac_coords = torch.from_numpy(np.array(r["frac_coords"], dtype=np.float32, copy=True))
    lattice = torch.from_numpy(np.array(r["lattice"], dtype=np.float32, copy=True))
    return (atom_fea, nbr_fea_t, nbr_fea_idx, poly_fea_t, poly_fea_idx, nbr_angle,
            frac_coords, lattice)


def _build_sample(data_row, max_num_nbr, max_num_poly_nbr,
                  use_poly_edges, use_bond_angles, build_angle_bias=False):
    """Build one fully-padded crystal sample from its graph JSON on disk.

    Module-level (not a method) so it is picklable by a ``spawn`` multiprocessing
    pool — the prebuild path fans this out across cores. Returns CPU tensors in the
    exact tuple shape __getitem__ yields. Pure CPU/numpy work — never touches CUDA
    — so it is safe to run in spawned workers. Now a thin composition of
    _extract_ragged + _assemble_sample (shared with the packed-dataset pipeline).
    """
    cif_id, target, graph_path, label = data_row

    graph = CIFDataV4._read_graph(graph_path)
    ragged = _extract_ragged(graph)
    # Pass the FULL assemble tuple through (now includes frac_coords + lattice);
    # collate_pool slices [:6], collate_pool_geom uses [6:]. Must match the packed
    # backend's __getitem__, which also returns the whole _assemble_sample tuple.
    sample = _assemble_sample(ragged, max_num_nbr, max_num_poly_nbr,
                              use_poly_edges, use_bond_angles, build_angle_bias)

    target = torch.FloatTensor([float(target)])
    label = torch.LongTensor([int(label)])
    return (sample, target, label, cif_id)


def _sample_to_device(sample, device):
    """Move a built sample's tensors onto ``device`` (the cif_id string is left as-is).
    Arity-agnostic over the input tuple so it handles the geometry-carrying sample."""
    sample_in, target, label, cif_id = sample
    return (
        tuple(t.to(device) for t in sample_in),
        target.to(device), label.to(device), cif_id,
    )


class CIFDataV4(Dataset):
    """Dataset that loads pre-computed crystal_graph_v4 JSON files.

    Parameters
    ----------
    index_path : str
        Path to the index pickle produced by generate_CGv4_DB.
    max_num_nbr : int
        Bonding neighbor list will be padded/truncated to this length.
    max_num_poly_nbr : int
        Polyhedral neighbor list will be padded/truncated to this length.
    graph_cache_size : int
        Max number of fully-built crystal samples to keep in the per-process LRU
        cache. Bounds memory when the dataset is large (e.g. the ~55k non-SC
        pool): with DataLoader workers each worker holds up to this many samples.
        Set 0 (or negative) for an unbounded cache.
    random_seed : int
    target_column : str, optional
        Name of the index column to use as the regression target (for multi-
        target indexes, e.g. 'formation_energy_per_atom' or 'e_above_hull').
        Defaults to the legacy 'value' column, then 'tc'. Rows where the chosen
        target is NaN are dropped.
    """

    def __init__(self, index_path: str, max_num_nbr: int = 14,
                 max_num_poly_nbr: int = 16, graph_cache_size: int = 4096,
                 random_seed: int = 123, target_column: str = None,
                 use_bond_angles: bool = False, use_poly_edges: bool = True,
                 build_angle_bias: bool = False):
        assert os.path.exists(index_path), f"Index file not found: {index_path}"
        # When True, __getitem__ also emits a per-atom (M, M) cos-angle matrix for
        # the set-transformer aggregation's pairwise attention bias (idea A), built
        # from the graph's angle_triplets. Independent of use_bond_angles (which
        # concatenates a mean-RBF angle vector onto each edge instead).
        self.build_angle_bias = build_angle_bias
        # When False, skip building the polyhedral-edge tensors entirely: the model
        # ignores poly inputs in that case (MPNNModel.forward guards on
        # use_poly_edges), so the per-atom poly sort/pad/stack — ~80% of a built
        # sample's bytes and the bulk of __getitem__'s CPU cost when max_num_poly_nbr
        # is large — would be pure waste (built, cached, AND copied to the GPU each
        # batch). Disabled crystals get empty (N, 0, POLY_FEA_LEN) poly tensors,
        # which collate/transfer handle as no-ops. Must match the model's
        # use_poly_edges so the two agree on whether poly is used.
        self.use_poly_edges = use_poly_edges
        # When True, each bonding edge gets an RBF(cos angle) "angular environment"
        # vector (ANGLE_FEA_LEN dims) concatenated to its NBR_FEA_LEN base features, built
        # from the graph's stored bond-angle triplets. Requires graphs rebuilt with
        # the angle_triplets field; older graphs yield zero angle vectors.
        self.use_bond_angles = use_bond_angles

        index_df = pd.read_pickle(index_path)
        self.target_column = target_key = _select_target_key(
            index_df, target_column, index_path)

        # `label` (1 = SC, 0 = non-SC) defaults to 1 for older indexes without the
        # column, leaving the regression path unaffected. Rows whose chosen target
        # is missing (NaN — e.g. a source that didn't carry this target) are
        # dropped so they can't poison training.
        # Shared row construction (see build_data_rows — identical for both
        # backends so seeds map to identical splits): element 2 of each data
        # tuple is this backend's graph path; `groups` (parallel mp_id list)
        # enables material-level splits for trajectory datasets.
        self.data, self.groups, self.labels, dropped = build_data_rows(
            index_df, target_key, index_df["graph_path"].tolist(), random_seed)
        if dropped:
            print(f"CIFDataV4: dropped {dropped} rows with no '{target_key}' value")

        # Guard against the silent footgun: use_bond_angles=True on graphs built
        # before angle_triplets existed would yield all-zero angular features with
        # no error. The 'angle_triplets' key is always present in newly-built
        # graphs (even if empty); its absence means stale graphs -> fail loudly.
        if self.use_bond_angles and self.data:
            sample_graph = self._read_graph(self.data[0][2])
            if "angle_triplets" not in sample_graph:
                raise ValueError(
                    "use_bond_angles=True but the graphs have no 'angle_triplets' "
                    f"field (checked {self.data[0][2]}). Rebuild the cgv4 graphs with "
                    "the updated builder, or set use_bond_angles=false.")

        self.max_num_nbr = max_num_nbr
        self.max_num_poly_nbr = max_num_poly_nbr
        # Bounded (LRU) in-memory cache of fully-built samples, keyed by index.
        # The graphs are immutable and __getitem__ is deterministic, so we cache
        # its *output* (the built tensors) rather than the parsed JSON: this skips
        # not just the ~135 KB read + json.load but ALL the per-atom neighbor
        # sorting / padding / stacking on every epoch after the first — that Python
        # work is the dominant CPU cost and the usual input-pipeline bottleneck.
        # Built tensors are also smaller than the parsed JSON, so the cache is
        # cheaper than the old graph cache. The cap bounds memory for large
        # datasets (each DataLoader worker holds its own cache). Lazily filled per
        # process; use persistent_workers so each worker's cache survives epochs.
        self.graph_cache_size = graph_cache_size
        self._item_cache = OrderedDict()
        # When set by prebuild(), holds every sample fully built (optionally on the
        # GPU); __getitem__ then just indexes it and the lazy/LRU path is bypassed.
        self._prebuilt = None

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _read_graph(graph_path):
        """Parse a graph JSON from disk (no caching).

        Used on a cold __getitem__ miss and by compute_feature_stats. We do NOT
        cache the parsed JSON: __getitem__ caches its built output instead (see
        self._item_cache), and feature-stats reads each graph exactly once.
        """
        with open(graph_path) as f:
            return json.load(f)

    def prebuild(self, device="cpu", progress_every=5000):
        """Build every sample once, up front, and keep them all resident.

        Eliminates the per-epoch JSON-read + sort/pad/stack entirely: after this,
        __getitem__ is a list index. The whole built dataset is small (tens of KB
        per crystal), so a single copy fits in RAM — or, with ``device='cuda'``,
        straight in VRAM, which also removes the per-batch host->device copy
        (batches just gather already-resident tensors).

        Deliberately SINGLE-PROCESS. A multiprocessing pool was measured ~1.6x
        SLOWER here (288s vs ~184s for ~49k graphs): the build is fast per item
        (~4ms), so wall time is dominated by shipping the ~2-9 GB of built tensors
        back from workers through pipes — that pickle IPC outweighs the parallelism,
        and it also doubles peak RAM. Spawn/fork would add CUDA-context hazards on
        top. A one-time ~3-min build is negligible against a multi-hour training run.
        Each sample is moved to ``device`` as it's built, so peak CPU RAM stays at
        one crystal regardless of dataset size.

        Because the dataset is shared by the train/val/test loaders, every index is
        prebuilt. After this call num_workers MUST be 0 (CUDA tensors can't cross a
        DataLoader worker boundary, and there's no build work left to parallelize).
        """
        dev = torch.device(device)
        n = len(self.data)
        built = []
        for i, rec in enumerate(self.data):
            sample = _build_sample(rec, self.max_num_nbr, self.max_num_poly_nbr,
                                   self.use_poly_edges, self.use_bond_angles,
                                   self.build_angle_bias)
            if dev.type != "cpu":
                sample = _sample_to_device(sample, dev)
            built.append(sample)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"   prebuilt {i + 1}/{n} samples", flush=True)

        self._prebuilt = built
        # The lazy LRU cache is now dead weight; drop it and disable the lazy path.
        self._item_cache = OrderedDict()
        return self

    def __getitem__(self, idx):
        # Fast path: everything was prebuilt (optionally already on the GPU), so a
        # sample is just a list index — no disk read, no rebuild, no transfer.
        if self._prebuilt is not None:
            return self._prebuilt[idx]

        # Serve the fully-built sample from the per-process LRU cache if present.
        # __getitem__ is deterministic (immutable graphs), so caching the built
        # tensors skips all the neighbor sorting / padding / stacking — the dominant
        # CPU cost — on every epoch after the first. collate never mutates these
        # tensors, so returning the same objects across epochs is safe.
        cached = self._item_cache.get(idx)
        if cached is not None:
            self._item_cache.move_to_end(idx)   # mark most-recently-used
            return cached

        result = _build_sample(self.data[idx], self.max_num_nbr,
                               self.max_num_poly_nbr, self.use_poly_edges,
                               self.use_bond_angles, self.build_angle_bias)

        # Store the built sample for reuse on later epochs (LRU-bounded).
        self._item_cache[idx] = result
        if self.graph_cache_size and len(self._item_cache) > self.graph_cache_size:
            self._item_cache.popitem(last=False)   # evict least-recently-used
        return result


def resolve_split_by(configured, dataset):
    """Resolve the train/val/test split granularity.

    Single source of truth for the default, shared by training (MPNNMain) and
    re-evaluation (scripts/eval_test.py) so the two can never drift: an explicit
    config value wins; otherwise material-level grouping whenever the dataset
    carries group info (trajectory datasets — frame-level random splits leak
    near-duplicate frames), else the historical frame/row split.
    """
    if configured is not None:
        return configured
    return "material" if getattr(dataset, "groups", None) is not None else "frame"


def load_cif_dataset(index_path, **kwargs):
    """Open a training dataset from either backend, keyed on what the path is:

    - a pack directory (contains pack_header.json) -> PackedCIFDataV4, the
      columnar memmap store written by MPNNPack.pack_dataset (no JSON parse or
      neighbor build at read time);
    - anything else -> the classic CIFDataV4 over an index pickle + graph JSONs.

    Both expose the same interface (data/labels/groups/target_column, identical
    sample tuples), so callers never branch again after this point.
    """
    if os.path.isdir(index_path) and os.path.exists(
            os.path.join(index_path, "pack_header.json")):
        from pack import PackedCIFDataV4
        return PackedCIFDataV4(index_path, **kwargs)
    return CIFDataV4(index_path=index_path, **kwargs)


def collate_pool(dataset_list):
    """Collate crystals into a batch, offsetting neighbor indices into the batch
    atom array.

    The crystal membership travels as ONE segment tensor ``crystal_seg`` (N,)
    mapping each atom to its crystal index, plus the python int ``n_crystals``
    (kept CPU-side so the model never needs a .max().item() sync). The previous
    list-of-arange-tensors representation cost one tiny tensor + one host->device
    copy PER CRYSTAL PER BATCH (~128 of each) and forced Python loops in the
    pooling readout.
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_poly_fea, batch_poly_fea_idx, batch_nbr_angle = [], [], []
    counts, batch_target, batch_label = [], [], []
    batch_cif_ids = []
    base_idx = 0
    for (sample, target, label, cif_id) in dataset_list:
        # sample[:6] are the model inputs; [6:] (frac_coords, lattice) are geometry
        # that only collate_pool_geom (GPS) consumes — sliced off here, so MPNN's
        # batch is unchanged.
        atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx, nbr_angle = sample[:6]
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        batch_poly_fea.append(poly_fea)
        batch_poly_fea_idx.append(poly_fea_idx + base_idx)
        batch_nbr_angle.append(nbr_angle)
        counts.append(n_i)
        batch_target.append(target)
        batch_label.append(label)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    n_crystals = len(dataset_list)
    device = batch_atom_fea[0].device   # samples' own device (GPU when prebuilt-cuda)
    crystal_seg = torch.repeat_interleave(
        torch.arange(n_crystals, device=device),
        torch.tensor(counts, device=device))

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        torch.cat(batch_poly_fea, dim=0),
        torch.cat(batch_poly_fea_idx, dim=0),
        torch.cat(batch_nbr_angle, dim=0),
        crystal_seg,
        n_crystals,
    ), torch.stack(batch_target, dim=0), torch.cat(batch_label, dim=0), batch_cif_ids


def collate_pool_geom(dataset_list):
    """collate_pool + geometry for the GPS distance bias: appends batch-concatenated
    fractional coords (N, 3) and per-crystal lattice (B, 3, 3) to the input tuple,
    in the SAME atom/crystal order collate_pool uses. The GPS loader selects this;
    everything else keeps collate_pool (so the MPNN batch is untouched)."""
    base_input, targets, labels, cif_ids = collate_pool(dataset_list)
    fracs = [sample[6] for (sample, *_rest) in dataset_list]    # each (n_i, 3)
    lats = [sample[7] for (sample, *_rest) in dataset_list]     # each (3, 3)
    frac_coords = torch.cat(fracs, dim=0)                       # (N, 3)
    lattice = torch.stack(lats, dim=0)                          # (B, 3, 3)
    return base_input + (frac_coords, lattice), targets, labels, cif_ids


def get_train_val_test_loader(dataset, collate_fn=default_collate,
                              batch_size=64, train_ratio=None,
                              val_ratio=0.1, test_ratio=0.1, return_test=False,
                              num_workers=0, pin_memory=False, **kwargs):
    total_size = len(dataset)
    if train_ratio is None:
        assert val_ratio + test_ratio < 1
        train_ratio = 1 - val_ratio - test_ratio
    else:
        assert train_ratio + val_ratio + test_ratio <= 1

    indices = list(range(total_size))
    train_size = kwargs.get("train_size") or int(train_ratio * total_size)
    test_size = kwargs.get("test_size") or int(test_ratio * total_size)
    valid_size = kwargs.get("val_size") or int(val_ratio * total_size)

    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size):-test_size])

    # Keep workers alive between epochs so each worker's built-sample cache
    # (CIFDataV4._item_cache) survives instead of being rebuilt every epoch.
    persistent = num_workers > 0

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin_memory, persistent_workers=persistent)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=pin_memory, persistent_workers=persistent)

    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=num_workers, collate_fn=collate_fn,
                                 pin_memory=pin_memory, persistent_workers=persistent)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


class BalancedEpochSampler(Sampler):
    """Class-balancing sampler for the SC/non-SC classifier.

    Each epoch yields all minority (superconductor) training indices plus a fresh
    random sample of ``n_majority`` majority (non-SC) indices, shuffled together.
    DataLoader calls ``iter(sampler)`` once per epoch, so the majority subset is
    re-drawn every epoch.
    """

    def __init__(self, minority_indices, majority_indices, n_majority, seed=123):
        self.minority = list(minority_indices)
        self.majority = list(majority_indices)
        self.n_majority = min(n_majority, len(self.majority))
        self._rng = random.Random(seed)

    def __iter__(self):
        sampled_major = self._rng.sample(self.majority, self.n_majority)
        epoch_indices = self.minority + sampled_major
        self._rng.shuffle(epoch_indices)
        return iter(epoch_indices)

    def __len__(self):
        return len(self.minority) + self.n_majority


def dataset_atom_counts(dataset):
    """Per-index crystal atom counts (aligned to dataset index), cheaply, or None.

    Packed store: index the in-memory ``n_atoms`` offset column at each row's meta
    position (``dataset.data[i][2]`` is that position for the packed backend), so
    no graphs are read. The lazy CIF backend has no such table -> None, and the
    caller falls back to plain batching.
    """
    off = getattr(dataset, "_off", None)
    if off is None or "n_atoms" not in off:
        return None
    n_atoms = np.asarray(off["n_atoms"])
    positions = np.fromiter((rec[2] for rec in dataset.data),
                            dtype=np.int64, count=len(dataset.data))
    return n_atoms[positions]


class SizeGroupedBatchSampler(Sampler):
    """Batch crystals of similar size so the model's padded within-crystal global
    attention (a ``(B, Lmax, d)`` tensor, ``Lmax`` = the largest cell in the batch)
    isn't blown up by one big cell sitting among small ones.

    Megabatch-sort bucketing: take the wrapped sampler's per-epoch index order,
    sort within windows of ``pool_factor * batch_size``, cut into batches, then
    shuffle the batch ORDER so size isn't monotonic across the epoch. Wrapping a
    per-epoch sampler (e.g. BalancedEpochSampler) preserves its re-sample/shuffle.

    Two caps make a batch:
    - ``batch_size``: max crystals per batch (the small-cell regime), and
    - ``max_atoms`` (optional): max summed atoms per batch. Because a size-grouped
      batch has little padding waste (all cells ~Lmax), summed atoms ~= B*Lmax, so
      this cap gives LARGE-cell batches FEWER crystals — which is what actually
      bounds the PEAK B*Lmax^2 attention cost (size-grouping alone only lowers the
      average). None disables it (fixed batch_size, average-only benefit).
    """

    def __init__(self, sampler, sizes, batch_size, max_atoms=None,
                 pool_factor=20, seed=123, drop_last=False):
        self.sampler = sampler
        self.sizes = sizes                       # array: dataset index -> atom count
        self.batch_size = int(batch_size)
        self.max_atoms = int(max_atoms) if max_atoms else None
        self.pool_factor = max(1, int(pool_factor))
        self.drop_last = drop_last
        self._rng = random.Random(seed)
        # The max_atoms cap makes batches variable-size, so the batch COUNT can't be
        # derived from len(sampler) alone. Materialize the epoch's batches once and
        # share them between __len__ and __iter__ so they always agree — the trainer
        # uses len(loader) for warmup_steps and the global step (epoch*len + i); a
        # len that disagreed with the real batch count would corrupt that schedule.
        self._pending = None

    def _cut(self, window):
        batch, batch_atoms = [], 0
        for i in window:
            s = int(self.sizes[i])
            full = len(batch) >= self.batch_size
            over = self.max_atoms is not None and batch and batch_atoms + s > self.max_atoms
            if full or over:
                yield batch
                batch, batch_atoms = [], 0
            batch.append(i)
            batch_atoms += s
        if batch:
            yield batch

    def _build(self):
        idxs = list(self.sampler)               # consumes the wrapped (per-epoch) sampler
        pool = self.pool_factor * self.batch_size
        batches = []
        for s in range(0, len(idxs), pool):
            window = sorted(idxs[s:s + pool], key=lambda i: self.sizes[i])
            for batch in self._cut(window):
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)
        self._rng.shuffle(batches)
        return batches

    def __len__(self):
        # Build (and cache) this epoch's layout so the count is exact; __iter__ then
        # consumes the same layout. Re-sampling the wrapped sampler happens here.
        if self._pending is None:
            self._pending = self._build()
        return len(self._pending)

    def __iter__(self):
        if self._pending is None:
            self._pending = self._build()
        batches = self._pending
        yield from batches
        # Clear only AFTER a full pass: a len(loader) call DURING iteration (the
        # trainer's per-step progress print) then returns this same cached layout
        # instead of rebuilding + re-consuming the wrapped sampler mid-epoch; the
        # next epoch sees _pending=None and rebuilds/re-samples.
        self._pending = None


def nonsc_count_for_ratio(n_sc_train, sc_to_nonsc_ratio, n_nonsc_available):
    """Number of non-SC samples to draw per epoch for a given SC:non-SC ratio.

    ``sc_to_nonsc_ratio`` is SC count / non-SC count:
      - inf (or None / <= 0): no non-SC used (n_nonsc = 0). This is the default.
      - 1.0:   one non-SC per SC (n_nonsc == n_sc_train).
      - 2.0:   one non-SC per two SC (n_nonsc == n_sc_train / 2).
    Capped at the number of non-SC actually available in the train split.
    """
    if sc_to_nonsc_ratio is None or sc_to_nonsc_ratio == float("inf") or sc_to_nonsc_ratio <= 0:
        return 0
    return min(round(n_sc_train / sc_to_nonsc_ratio), n_nonsc_available)


def get_sc_nonsc_loaders(dataset, batch_size=64, val_ratio=0.1, test_ratio=0.1,
                         sc_to_nonsc_ratio=float("inf"), num_workers=0,
                         pin_memory=False, seed=123, split_by="frame",
                         size_grouped=False, max_atoms_per_batch=None,
                         size_pool_factor=20, prefetch_factor=None,
                         collate_fn=collate_pool):
    """Build SC/non-SC loaders shared by the regression and classification tasks.

    Stratified per-class split into train/val/test. ``sc_to_nonsc_ratio`` governs
    the SC:non-SC composition of EVERY split (not just train): the train loader
    draws all SC-train indices plus a fresh ratio-sized non-SC sample each epoch
    (reshuffled via BalancedEpochSampler), and val/test get a fixed ratio-sized
    non-SC subset. val/test are still returned in a "realistic" form (the
    configured ratio) and a "balanced" form (1:1, drawn from the same ratio-limited
    pool); the two coincide when ratio >= 1.

    With the default ratio of inf, no non-SC are used in ANY split: train, val, and
    test are all SC-only (so non-SC entries never leak into the eval sets).

    Returns a dict: train, val_realistic, val_balanced, test_realistic,
    test_balanced, split_sizes, and the train-pool index lists
    (train_sc_idx, train_nonsc_idx) + n_nonsc_per_epoch for the caller's use
    (e.g. building the regression target normalizer).
    """
    labels = np.asarray(dataset.labels)
    rng = random.Random(seed)

    # split_by="material": split at the GROUP level (dataset.groups, e.g. the
    # parent mp_id of each trajectory frame) so all frames of one material land in
    # the same split. A frame-level random split on trajectory data leaks
    # near-duplicate frames of train materials into val/test, making eval metrics
    # optimistic. Explicitly requesting it without group info is an error — a
    # silent frame fallback would reintroduce exactly that leakage.
    groups = getattr(dataset, "groups", None)
    if split_by == "material" and groups is None:
        raise ValueError(
            "split_by='material' but the dataset index has no usable 'mp_id' "
            "column. Rebuild/re-index the dataset (build-mptrj re-run is enough — "
            "graphs are kept) or set split_by='frame'.")
    if split_by not in ("frame", "material"):
        raise ValueError(f"Unknown split_by '{split_by}' (use 'frame' or 'material').")

    def split_class(idx_list):
        idx = list(idx_list)
        if split_by == "material":
            # Shuffle materials, then greedily fill test -> val -> train with whole
            # materials until each split's FRAME budget is met. Frame counts only
            # approximate the ratios (a material's frames are indivisible).
            by_group = {}
            for i in idx:
                by_group.setdefault(groups[i], []).append(i)
            gkeys = list(by_group.keys())
            rng.shuffle(gkeys)
            n = len(idx)
            n_test, n_val = int(test_ratio * n), int(val_ratio * n)
            train, val, test = [], [], []
            for g in gkeys:
                block = by_group[g]
                if len(test) < n_test:
                    test.extend(block)
                elif len(val) < n_val:
                    val.extend(block)
                else:
                    train.extend(block)
            for part in (train, val, test):
                rng.shuffle(part)
            return train, val, test
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(test_ratio * n)
        n_val = int(val_ratio * n)
        return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]

    sc_train, sc_val, sc_test = split_class(np.where(labels == 1)[0].tolist())
    ns_train, ns_val, ns_test = split_class(np.where(labels == 0)[0].tolist())

    # Apply the SC:non-SC ratio to EVERY split, not just train. Previously val/test
    # included ALL non-SC ("realistic" = true imbalance), so even at ratio=inf the
    # eval sets still contained non-SC (e.g. Materials Project entries leaking into
    # the SC-only T_c-regression test set). Now the ratio governs composition
    # uniformly: ratio=inf => SC-only train/val/test; ratio=1.0 => 1:1 everywhere.
    n_nonsc_train = nonsc_count_for_ratio(len(sc_train), sc_to_nonsc_ratio, len(ns_train))
    n_nonsc_val = nonsc_count_for_ratio(len(sc_val), sc_to_nonsc_ratio, len(ns_val))
    n_nonsc_test = nonsc_count_for_ratio(len(sc_test), sc_to_nonsc_ratio, len(ns_test))

    # Fixed ratio-sized non-SC subset for val/test (train re-samples its own each
    # epoch via BalancedEpochSampler, so it keeps the full ns_train pool).
    ns_val_keep = rng.sample(ns_val, n_nonsc_val) if n_nonsc_val else []
    ns_test_keep = rng.sample(ns_test, n_nonsc_test) if n_nonsc_test else []

    # Optional size-grouped batching to bound the GPS global-attention memory
    # (B x Lmax^2). Cheap only for the packed backend; otherwise fall back.
    sizes = dataset_atom_counts(dataset) if size_grouped else None
    if size_grouped and sizes is None:
        warnings.warn("size_grouped batching requested but this dataset backend "
                      "exposes no cheap atom counts; using plain batching.",
                      RuntimeWarning)

    # Shared DataLoader kwargs. prefetch_factor (workers stay this many batches
    # ahead) is only valid with workers, and smooths the variable-size batches the
    # size-grouped sampler emits; omit it for the single-process path.
    loader_kw = dict(num_workers=num_workers, collate_fn=collate_fn,
                     pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    if prefetch_factor and num_workers > 0:
        loader_kw["prefetch_factor"] = prefetch_factor

    def make_loader(sampler):
        if sizes is not None:
            batch_sampler = SizeGroupedBatchSampler(
                sampler, sizes, batch_size, max_atoms=max_atoms_per_batch,
                pool_factor=size_pool_factor, seed=seed)
            return DataLoader(dataset, batch_sampler=batch_sampler, **loader_kw)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, **loader_kw)

    def balanced_subset(min_idx, maj_idx):
        k = min(len(min_idx), len(maj_idx))
        return list(min_idx) + rng.sample(maj_idx, k)

    train_loader = make_loader(
        BalancedEpochSampler(sc_train, ns_train, n_nonsc_train, seed=seed))

    return {
        "train": train_loader,
        # "realistic" now reflects the configured ratio (not the dataset's true
        # imbalance); "balanced" remains a 1:1 diagnostic drawn from the same
        # ratio-limited non-SC pool, so the two coincide when ratio >= 1 and both
        # are SC-only when ratio=inf.
        "val_realistic": make_loader(SubsetRandomSampler(sc_val + ns_val_keep)),
        "val_balanced": make_loader(SubsetRandomSampler(balanced_subset(sc_val, ns_val_keep))),
        "test_realistic": make_loader(SubsetRandomSampler(sc_test + ns_test_keep)),
        "test_balanced": make_loader(SubsetRandomSampler(balanced_subset(sc_test, ns_test_keep))),
        "train_sc_idx": sc_train,
        "train_nonsc_idx": ns_train,
        "n_nonsc_per_epoch": n_nonsc_train,
        "split_sizes": {
            "train_sc": len(sc_train), "train_nonsc": len(ns_train),
            "train_nonsc_per_epoch": n_nonsc_train,
            "val_sc": len(sc_val), "val_nonsc": len(ns_val_keep),
            "test_sc": len(sc_test), "test_nonsc": len(ns_test_keep),
        },
    }
