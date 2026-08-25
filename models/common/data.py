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

# Base-block columns derived from the RELAXED GEOMETRY (ecn_value, shannon_radius,
# cn_core, hist_corner/edge/face/other). mask_geometry_features zeroes exactly these,
# leaving the composition-only columns (Z, oxidation, ion_role, chi x2, IE, EA) —
# the "structure-independent encoder input" rung of the no-pretrain ladder.
GEO_FEATURE_COLS = slice(5, 12)

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
# Dihedrals (4-body torsions) are encoded with the SAME cos-RBF basis as angles
# (cos(dihedral) is in [-1, 1] too). See dihedral_node_features.
DIHEDRAL_FEA_LEN = ANGLE_FEA_LEN
# Multitask DOS target: per-structure total density of states sampled on a FIXED energy
# grid (E_F-aligned). The fetch (Download_MP_dos) and the model's `n_energy` must match this.
# Narrow +/-1 eV window (128 bins) — see Download_MP_dos: concentrates the DOS loss on the
# SC-relevant near-E_F states instead of the wide -10..+5 eV window.
DOS_N_ENERGY = 128


def _angle_rbf(cos_vals) -> np.ndarray:
    """Gaussian RBF expansion of cos(theta) values -> (len(cos_vals), ANGLE_FEA_LEN)."""
    c = np.asarray(cos_vals, dtype=np.float32).reshape(-1, 1)
    diff = c - ANGLE_RBF_CENTERS.reshape(1, -1)
    return np.exp(-(diff ** 2) / (2.0 * ANGLE_RBF_WIDTH ** 2)).astype(np.float32)


def dihedral_node_features(graph, n_atoms):
    """Per-atom mean Gaussian-RBF of cos(dihedral) over every 4-body torsion whose
    CENTRAL bond is incident to the atom (i.e. the atom is a torsion-axis endpoint).
    Returns (n_atoms, DIHEDRAL_FEA_LEN). A coarse per-NODE summary of the torsional
    environment -- the per-bond form is the refinement. Legacy graphs without a
    'dihedrals' field yield zeros. Shared by the packer (_extract_ragged) and the
    lazy stats path (compute_feature_stats) so the two representations can't drift."""
    out = np.zeros((n_atoms, DIHEDRAL_FEA_LEN), dtype=np.float32)
    dihedrals = graph.get("dihedrals", [])
    if not dihedrals or not n_atoms:
        return out
    edges_by_id = {e["id"]: e for e in graph.get("edges", [])}
    axes, cosv = [], []
    for d in dihedrals:                        # d = [central_edge, edge_i, edge_l, cos]
        e = edges_by_id.get(int(d[0]))
        if e is not None:
            axes.append((int(e["source"]), int(e["target"])))
            cosv.append(float(d[3]))
    if not cosv:
        return out
    rbf = _angle_rbf(cosv)                      # (D, DIHEDRAL_FEA_LEN)
    atoms = np.asarray(axes, dtype=np.int64)    # (D, 2): the torsion-axis endpoints
    cnt = np.zeros(n_atoms, dtype=np.int64)
    for col in (0, 1):
        a = atoms[:, col]
        ok = (a >= 0) & (a < n_atoms)
        np.add.at(out, a[ok], rbf[ok])          # scatter-add onto both axis endpoints
        np.add.at(cnt, a[ok], 1)
    nz = cnt > 0
    out[nz] /= cnt[nz, None]                     # mean over incident torsions
    return out


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


# Physically-motivated per-element features (mass -> phonons; group/row -> electronic
# structure & size; # unpaired electrons -> magnetic-moment proxy / spin leg). Added
# at ASSEMBLE time from the stored Z (atom_fea[:, 0]) when use_rich_node_features is
# on, so no re-pack is needed; standardized via compute_feature_stats like the rest.
RICH_NODE_FEA_LEN = 4
_RICH_L = {"s": 0, "p": 1, "d": 2, "f": 3}


def _rich_node_features_for_Z(Z):
    e = PmgElement.from_Z(int(Z))
    n_unpaired = 0
    for _n, l, occ in e.full_electronic_structure:        # Hund's rule per subshell
        g = 2 * _RICH_L[l] + 1
        n_unpaired += occ if occ <= g else 2 * g - occ
    return [float(e.atomic_mass), float(e.group or 0), float(e.row or 0), float(n_unpaired)]


_RICH_TABLE = np.zeros((119, RICH_NODE_FEA_LEN), dtype=np.float32)
for _z in range(1, 119):
    try:
        _RICH_TABLE[_z] = _rich_node_features_for_Z(_z)
    except Exception:                                     # noqa: BLE001 -> zeros
        pass


# ---- valence SUBSHELL occupancy: element-AGNOSTIC orbital-character handle ----
# [n_s, n_p, n_d, n_f] of the ION at each site (occupancy-weighted oxidation). Encodes
# electronic CONFIGURATION, not element identity: Cu2+ and Ni1+ both read [0,0,9,0] (d9),
# while a d2 site [0,0,2,0] and a p2 site [2,2,0,0] stay distinct, and f-electron
# (heavy-fermion) systems are visible (Ce3+ = [0,0,0,1]). d/f-block CATIONS collapse
# valence into the (n-1)d / (n-2)f shell (Ni+ = 3d9, not 3d8 4s1). This LOAD-time
# computation from stored Z + oxidation_state is the LEGACY FALLBACK: builder schema
# >= v4.3 bakes a per-species occupancy-weighted valence block into the graph, which
# assemble prefers when present. Opt-in via use_valence_features;
# concat order [base | rich | valence | cf | dihedral], mirrored across every assembly site.
VALENCE_NODE_FEA_LEN = 4
# AOM crystal-field block baked by builder schema >= v4.3 (RPToleranceFactor
# crystal_field_aom): cf_levels[5] + cf_occ[5] + cf_frontier_gap + cf_unpaired.
# Baked-only (needs the builder's neighbor geometry + anion roles) — there is
# deliberately NO load-time fallback: use_cf_features on a cf-less graph/pack
# raises, so a masked union can't silently mix real and zero CF blocks.
CF_FEA_LEN = 12
# Bond-valence block (builder schema v4.4 / CF_SCHEMA 3): [bvs, bvs_mismatch].
# Baked-only, fail-loud like cf (see the CF_FEA_LEN note above).
BVS_FEA_LEN = 2
_L = {"s": 0, "p": 1, "d": 2, "f": 3}
_NOBLE_Z = [2, 10, 18, 36, 54, 86]


def _valence_orbitals(el):
    """{(n, l): valence occupancy} = neutral orbitals BEYOND the preceding noble-gas core
    (so Ce's 4f/5d/6s count as valence, its 1s-5p Xe core does not — robust across the
    Aufbau/Madelung ordering that a simple 'highest n' rule gets wrong for f-block)."""
    cfg = {(n, l): occ for (n, l, occ) in el.full_electronic_structure}
    prev = max([z for z in _NOBLE_Z if z < el.Z], default=0)
    noble = ({(n, l): occ for (n, l, occ) in PmgElement.from_Z(prev).full_electronic_structure}
             if prev else {})
    return {(n, l): occ - noble.get((n, l), 0)
            for (n, l), occ in cfg.items() if occ - noble.get((n, l), 0) > 0}


_VAL_NEUTRAL = np.zeros((119, 4), dtype=np.float32)   # neutral valence [s,p,d,f] per Z
_REMOVE_ORDER = [[] for _ in range(119)]              # l-indices in cation removal order
_GROUP_TABLE = np.zeros(119, dtype=np.float32)
_IS_DFBLOCK = np.zeros(119, dtype=bool)               # d-block: valence collapses into d
for _z in range(1, 119):
    try:
        _el = PmgElement.from_Z(_z)
        _vo = _valence_orbitals(_el)
        if _el.block in ("s", "p"):   # main group: filled (n-1)d / (n-2)f are inert core,
            _vo = {k: v for k, v in _vo.items() if k[1] in ("s", "p")}  # keep only ns/np
        for (_n, _l), _occ in _vo.items():
            _VAL_NEUTRAL[_z, _L[_l]] += _occ
        # remove ns/np before (n-1)d before (n-2)f -> sort valence orbitals by (n desc, l desc)
        _REMOVE_ORDER[_z] = [_L[_l] for (_n, _l), _occ
                             in sorted(_vo.items(), key=lambda kv: (kv[0][0], _L[kv[0][1]]), reverse=True)]
        _GROUP_TABLE[_z] = float(_el.group or 0)
        _IS_DFBLOCK[_z] = (_el.block == "d")
    except Exception:                                 # noqa: BLE001
        pass


def valence_node_features(z_array, ox_array):
    """(N,) Z + (N,) oxidation_state -> (N, 4): valence subshell occupancy [n_s,n_p,n_d,n_f]
    of the ion. d-block cations collapse valence into d (group-ox); main-group removes the
    outer np then ns; anions fill the outer p then s. Cu2+ ~ Ni1+ ~ [0,0,9,0]."""
    z = np.clip(np.asarray(z_array, dtype=int), 0, 118)
    ox = np.asarray(ox_array, dtype=np.float32)
    out = np.zeros((len(z), 4), dtype=np.float32)
    for i in range(len(z)):
        zi = int(z[i]); oxi = float(ox[i])
        if _IS_DFBLOCK[zi] and oxi > 0:               # TM cation: all valence electrons in d
            out[i, 2] = max(0.0, min(10.0, _GROUP_TABLE[zi] - oxi)); continue
        occ = _VAL_NEUTRAL[zi].copy()
        if oxi > 0:                                   # cation: strip in (n desc, l desc) order
            r = oxi
            for li in _REMOVE_ORDER[zi]:
                if r <= 1e-9: break
                t = min(occ[li], r); occ[li] -= t; r -= t
        elif oxi < 0:                                 # anion: fill outer p then s
            a = -oxi
            t = min(6.0 - occ[1], a); occ[1] += t; a -= t
            occ[0] += min(2.0 - occ[0], a)
        out[i] = occ
    return out


def rich_node_features(z_array):
    """(N,) atomic numbers -> (N, RICH_NODE_FEA_LEN) element features via a table."""
    z = np.clip(np.asarray(z_array, dtype=np.int64), 0, 118)
    return _RICH_TABLE[z]


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


# --- Phonon-spectrum targets (eph_a2f: Eliashberg alpha^2F; ph_dos: phonon DOS) ---
# One SHARED fixed grid for both, in THz: 128 bins over 0-60 THz covers the
# corpus (Cerqueira batch-a omega_max: median 8.7, p95 37.9, max 60.8 THz —
# hydrides carry the tail; rare >60 THz weight is clipped). Spectra are baked
# into graph JSONs as "a2f" / "ph_dos" keys (bin-averaged, see bin_spectrum),
# NaN-masked when absent — same flow as the electronic "dos" target.
PHONON_N_BINS = 128
PHONON_W_MAX_THZ = 60.0


def bin_spectrum(w_thz, y, n_bins=PHONON_N_BINS, w_max=PHONON_W_MAX_THZ):
    """Integral-preserving bin average of a spectrum onto the fixed grid:
    resample the piecewise-linear curve on a fine grid (zero outside the data
    range) and average within each bin, so sum(bins)*dw ~= the original
    integral and downstream lambda/omega_log integrals survive binning."""
    w_thz = np.asarray(w_thz, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    fine = np.linspace(0.0, w_max, n_bins * 32, endpoint=False) + w_max / (n_bins * 64)
    yf = np.interp(fine, w_thz, y, left=0.0, right=0.0)
    return yf.reshape(n_bins, 32).mean(axis=1).astype(np.float32)


# Per-node summary of incident polyhedral edges, for encoders WITHOUT a poly
# message-passing path (the no-pretrain ladder's identity mode): mean of the
# node's POLY_FEA_LEN edge features over its poly neighbors + log1p(count).
# Nodes with no poly edges emit zeros. Consumed via use_poly_node_summary,
# concat order [base | rich | valence | cf | bvs | poly_summary | dih].
POLY_SUMMARY_FEA_LEN = POLY_FEA_LEN + 1


def poly_node_summary(r) -> np.ndarray:
    """(n_atoms, POLY_SUMMARY_FEA_LEN) from a ragged extraction's per-node
    CSR poly lists (poly_cnt + flat poly_fea)."""
    n = int(r["n_atoms"])
    out = np.zeros((n, POLY_SUMMARY_FEA_LEN), dtype=np.float32)
    fea = np.asarray(r["poly_fea"], dtype=np.float32)
    p0 = 0
    for i in range(n):
        c = int(r["poly_cnt"][i])
        if c:
            out[i, :POLY_FEA_LEN] = fea[p0:p0 + c].mean(axis=0)
            out[i, POLY_FEA_LEN] = np.log1p(c)
        p0 += c
    return out


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


def _select_target_key(index_df, target_column, index_path, required=True):
    """Pick which column supplies the regression target. A multi-target index
    (e.g. the MP energy dataset) carries several named target columns;
    `target_column` (from the config) selects one. With it unset we fall back to
    the legacy `value` column, then `tc`, so older single-target indexes keep
    working unchanged. The fallback only accepts a column that actually has data
    — a tc-less index has an all-empty `value`, so this raises and tells the user
    to set `target_column` instead of silently regressing on the wrong target.
    Shared by CIFDataV4 (index pickle) and PackedCIFDataV4 (pack meta).
    mp_id is the grouping key for material splits, NOT a target.

    `required=False` (multitask masked-union members): an absent target_column
    returns None instead of raising — the scalar energy target is then NaN-masked
    for that pack (its other targets, e.g. dos, still train). A union member that
    genuinely lacks the configured scalar target (the DOS pack has no formation
    energy) must NOT crash the whole run, and must NOT have all its rows dropped."""
    structural_cols = {"id", "value", "graph_path", "label", "mp_id"}
    named_targets = [c for c in index_df.columns
                     if c not in structural_cols and not index_df[c].isna().all()]
    if target_column is not None:
        if target_column not in index_df.columns:
            if not required:
                return None
            raise ValueError(
                f"target_column '{target_column}' not found in index "
                f"{index_path}; available columns: {list(index_df.columns)}")
        return target_column
    target_key = next(
        (c for c in ("value", "tc")
         if c in index_df.columns and not index_df[c].isna().all()),
        None)
    if target_key is None:
        if not required:
            return None
        raise ValueError(
            f"No usable default target ('value'/'tc' absent or all-empty) "
            f"in index {index_path}. Set 'target_column' in the config to "
            f"one of {named_targets}.")
    return target_key


def build_data_rows(index_df, target_key, third_full, random_seed, keep_all=False):
    """Shared row construction for both dataset backends (CIFDataV4 and
    PackedCIFDataV4): drop target-NaN rows, attach label/mp_id, seed-shuffle.

    The two backends MUST stay bit-identical here — the same seed has to produce
    the same ordering (hence the same train/val/test splits) regardless of which
    backend serves the samples. ``third_full`` is the backend-specific element 2
    of each row tuple (graph paths for the lazy loader, meta row positions for
    the packed one), aligned with the FULL index_df.

    `keep_all=True` (multitask masked-union members): KEEP every row even when its
    scalar target is NaN/absent — the masked multitask loss zeroes a missing target
    instead of training on it, so a row with no scalar energy but a real dos label
    must survive (dropping it would silently empty a DOS-only pack). The scalar
    target is numeric-coerced ('' -> NaN -> masked); target_key=None -> all-NaN
    scalar. keep_all=False is the legacy single-target path, bit-identical to before.

    Returns (data, groups, labels, dropped):
      data   : [(id, target_value, third, label), ...] shuffled
      groups : parallel mp_id list, or None unless EVERY row has one
               (all-or-nothing: a partially-populated grouping key can't support
               a leakage-free material split)
      labels : [label, ...] aligned with data
    """
    if target_key is None:
        # Multitask member without the configured scalar target column: all-NaN
        # scalar (energy masked), keep every row. (keep_all is implied.)
        target_vals = pd.Series([float("nan")] * len(index_df), index=index_df.index)
        valid = pd.Series(True, index=index_df.index)
    elif keep_all:
        target_vals = pd.to_numeric(index_df[target_key], errors="coerce")
        valid = pd.Series(True, index=index_df.index)
    else:
        target_vals = index_df[target_key]
        valid = target_vals.notna()
    dropped = int((~valid).sum())
    keep = valid.tolist()
    sub = index_df.loc[valid]
    third = [t for t, v in zip(third_full, keep) if v]
    labels = (sub["label"].astype(int).tolist()
              if "label" in sub.columns else [1] * len(sub))
    if "mp_id" in index_df.columns:
        mp_ids = [None if pd.isna(g) else g for g in sub["mp_id"].tolist()]
    else:
        mp_ids = [None] * len(sub)
    rows = list(zip(sub["id"].tolist(), target_vals.loc[valid].tolist(), third,
                    labels, mp_ids))
    random.seed(random_seed)
    random.shuffle(rows)
    data = [r[:4] for r in rows]
    group_vals = [r[4] for r in rows]
    groups = group_vals if group_vals and all(g is not None for g in group_vals) else None
    return data, groups, [rec[3] for rec in data], dropped


def subsample_frames_by_group(data, groups, labels, stride):
    """Keep every ``stride``-th frame per material (group), in trajectory order, to
    thin near-duplicate consecutive MPtrj frames — a near-linear epoch speedup for
    pretraining (the downstream metric is T_c transfer, not held-out frame MAE, so the
    full ~1.5M-frame trajectory density is overkill).

    NO-OP when ``stride<=1`` or there are no groups (without a material key we can't
    tell which rows are frames of the same trajectory, so we'd risk a non-material
    subset). Single-frame materials (the relaxed-MP DOS pack) keep their one frame, so
    applying this to a masked union only thins the trajectory member. Ordering proxy is
    each row's element-2 (packed: meta row position == pack/trajectory order; lazy:
    graph path), so the kept frames are evenly spread across the trajectory rather than
    clustered. Only DROPS rows (the surviving rows keep their post-shuffle order), so
    splits/feature-stats see a strict subset — material membership is unchanged, so a
    material-level split stays leakage-free. Returns filtered (data, groups, labels)."""
    if not stride or stride <= 1 or groups is None:
        return data, groups, labels
    by_group = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    keep = set()
    for idxs in by_group.values():
        ordered = sorted(idxs, key=lambda i: data[i][2])   # trajectory order
        keep.update(ordered[::stride])
    kept = [i for i in range(len(data)) if i in keep]       # preserve shuffled order
    return ([data[i] for i in kept], [groups[i] for i in kept],
            [labels[i] for i in kept])


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

    use_rich = getattr(dataset, "use_rich_node_features", False)
    use_val = getattr(dataset, "use_valence_features", False)
    use_cf = getattr(dataset, "use_cf_features", False)
    use_bvs = getattr(dataset, "use_bvs_features", False)
    use_dih = getattr(dataset, "use_dihedrals", False)
    mask_ox = getattr(dataset, "mask_oxidation_feature", False)
    mask_geo = getattr(dataset, "mask_geometry_features", False)
    use_psum = getattr(dataset, "use_poly_node_summary", False)
    node_rows, edge_rows, poly_rows = [], [], []
    zero_ang = np.zeros(ANGLE_FEA_LEN, dtype=np.float32)
    for i in idx:
        graph = dataset._read_graph(dataset.data[i][2])
        dih = dihedral_node_features(graph, len(graph["nodes"])) if use_dih else None
        # Same ragged extraction as sample assembly, so the summary can't drift.
        psum = poly_node_summary(_extract_ragged(graph)) if use_psum else None
        for ni, n in enumerate(graph["nodes"]):
            feat = _node_to_fea(n)
            if mask_ox:
                feat[1] = 0.0
            if mask_geo:
                feat[GEO_FEATURE_COLS] = 0.0
            if use_rich:        # match the assemble-time concat order: [base | rich | val | cf | dih]
                feat = np.concatenate([feat, rich_node_features([n["Z"]])[0]])
            if use_val:
                feat = np.concatenate([feat, (np.asarray(n["valence"], dtype=np.float32)
                                              if "valence" in n else
                                              valence_node_features([n["Z"]], [feat[1]])[0])])
            if use_cf:
                if "cf" not in n:
                    raise ValueError("use_cf_features=True but graph nodes carry no baked "
                                     "cf block — rebuild with builder schema >= v4.3.")
                feat = np.concatenate([feat, np.asarray(n["cf"], dtype=np.float32)])
            if use_bvs:
                if "bvs" not in n:
                    raise ValueError("use_bvs_features=True but graph nodes carry no "
                                     "baked bvs block — rebuild with schema >= v4.4.")
                feat = np.concatenate([feat, np.asarray([n["bvs"], n.get("bvs_mismatch", 0.0)],
                                                        dtype=np.float32)])
            if use_psum:
                feat = np.concatenate([feat, psum[ni]])
            if use_dih:
                feat = np.concatenate([feat, dih[ni]])
            node_rows.append(feat)
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
    node_dim = (NODE_FEA_LEN + (RICH_NODE_FEA_LEN if use_rich else 0)
                + (VALENCE_NODE_FEA_LEN if use_val else 0)
                + (CF_FEA_LEN if use_cf else 0)
                + (BVS_FEA_LEN if use_bvs else 0)
                + (POLY_SUMMARY_FEA_LEN if use_psum else 0)
                + (DIHEDRAL_FEA_LEN if use_dih else 0))
    return {
        "node": rows_meanstd(node_rows, node_dim),
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
    bond_nbr, bond_fea, slot_eids_all, bond_jimage = [], [], [], []
    for atom_i in range(n_atoms):
        neighbors = adjacency.get(str(atom_i), [])

        def _bond_len(pair):
            e = edges_by_id.get(pair[0])
            return e["bond_length"] if e is not None else 1e9

        eids = []
        seen_eid = {}
        for eid, nbr_id in sorted(neighbors, key=_bond_len):
            edge = edges_by_id.get(eid)
            if edge is None:
                continue
            cis = _center_is_source(edge, atom_i, nbr_id)
            bond_fea.append(_edge_to_fea(edge, cis))
            bond_nbr.append(nbr_id)
            # Per-edge periodic image of the NEIGHBOR relative to the CENTER, oriented
            # center->neighbor (negate the stored to_jimage when the center is the
            # target). Lets the model reconstruct the EXACT PBC bond vector
            # r_j + jimage@lattice - r_i for differentiable/conservative forces.
            # Absent (legacy graphs) -> (0,0,0) -> min-image fallback in the model.
            ji = edge.get("to_jimage")
            ji = (np.zeros(3, dtype=np.int16) if ji is None
                  else np.asarray(ji, dtype=np.int16))
            ji = ji if cis else (-ji).astype(np.int16)
            # A self-image edge (source==target) occupies TWO slots of this center's
            # list with the SAME edge id (both have cis=True). They are the +image and
            # -image of the same bond, so the 2nd occurrence must flip sign — else both
            # point the same way, double-counting one image and dropping the other
            # (wrong PBC bond vector / forces). Only self-image edges duplicate an eid
            # within one center's list, so the occurrence count is a sufficient test.
            occ = seen_eid.get(eid, 0)
            seen_eid[eid] = occ + 1
            if occ:
                ji = (-ji).astype(np.int16)
            bond_jimage.append(ji)
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

    # Per-atom 4-body torsion summary (atom-aligned, reuses atom_start/n_atoms in the
    # pack like frac_coords); concatenated onto the node vector at assemble time when
    # use_dihedrals. Legacy graphs -> all zeros.
    dih_node = dihedral_node_features(graph, n_atoms)

    # Baked electronic-structure blocks (builder schema >= v4.3): per-node valence
    # subshells (4) + AOM crystal-field block (12). Absent from legacy graphs ->
    # None, so assemble falls back (valence: load-time computation) or raises
    # (cf: baked-only). The pack writer zero-fills None for its uniform bins and
    # records has_* header flags so the pack reader restores the None semantics.
    def _baked_block(key, width):
        stamped = sum(key in n for n in graph["nodes"])
        if stamped == 0:
            return None
        if stamped != n_atoms:
            # all-or-none: a partially-stamped graph (corruption/hand-edit — no
            # current writer produces one) must fail loudly, not silently mix
            # real rows with zero-filled ones (review 2026-07-28).
            raise ValueError(f"graph has '{key}' on {stamped}/{n_atoms} nodes — "
                             "corrupt or partially-baked; rebuild it")
        return np.stack([np.asarray(n[key], dtype=np.float32)
                         for n in graph["nodes"]])
    valence_baked = _baked_block("valence", VALENCE_NODE_FEA_LEN)
    cf_baked = _baked_block("cf", CF_FEA_LEN)
    # bvs is stored as two scalar node keys; assemble into the 2-wide block
    if any("bvs" in n for n in graph["nodes"]):
        stamped = sum("bvs" in n for n in graph["nodes"])
        if stamped != n_atoms:
            raise ValueError(f"graph has 'bvs' on {stamped}/{n_atoms} nodes — "
                             "corrupt or partially-baked; rebuild it")
        bvs_baked = np.array([[n["bvs"], n.get("bvs_mismatch", 0.0)]
                              for n in graph["nodes"]], dtype=np.float32)
    else:
        bvs_baked = None

    # Multi-task TARGET tensors carried alongside the inputs: per-atom forces (N,3) +
    # magmom (N,1), per-structure stress (3,3). Absent in a graph -> NaN sentinel, so
    # the masked multi-task loss reads presence via isfinite and trains only real labels
    # (NaN round-trips through the float32 pack; never seen by the model, only the loss).
    def _atom_target(key, width):
        v = graph.get(key)
        if v is None:
            return np.full((n_atoms, width), np.nan, dtype=np.float32)
        a = np.asarray(v, dtype=np.float32).reshape(-1, width)
        return a if a.shape[0] == n_atoms else np.full((n_atoms, width), np.nan, dtype=np.float32)
    forces = _atom_target("forces", 3)
    magmom = _atom_target("magmom", 1)
    st = graph.get("stress")
    stress = (np.asarray(st, dtype=np.float32).reshape(3, 3)
              if st is not None else np.full((3, 3), np.nan, dtype=np.float32))
    # Per-structure total DOS target (DOS_N_ENERGY,) on the fixed E_F-aligned grid; absent
    # (MPtrj frames / non-DOS materials) -> NaN -> masked. The DOS-bearing population is the
    # relaxed MP graphs (a separate masked-union member of the multitask training set).
    # A PRESENT dos must be exactly DOS_N_ENERGY long (reshape raises otherwise — loud, like
    # stress; a silent NaN would zero out a dataset's DOS labels on a stale-grid mismatch).
    dv = graph.get("dos")
    dos = (np.asarray(dv, dtype=np.float32).reshape(DOS_N_ENERGY)
           if dv is not None else np.full(DOS_N_ENERGY, np.nan, dtype=np.float32))
    # Phonon-spectrum targets on the shared PHONON grid (see bin_spectrum);
    # absent -> NaN -> masked, exactly like dos. Loud reshape on width mismatch.
    av = graph.get("a2f")
    a2f = (np.asarray(av, dtype=np.float32).reshape(PHONON_N_BINS)
           if av is not None else np.full(PHONON_N_BINS, np.nan, dtype=np.float32))
    pv = graph.get("ph_dos")
    ph_dos = (np.asarray(pv, dtype=np.float32).reshape(PHONON_N_BINS)
              if pv is not None else np.full(PHONON_N_BINS, np.nan, dtype=np.float32))

    return {
        "n_atoms": n_atoms,
        "atom_fea": atom_fea.astype(np.float32),
        "frac_coords": frac_coords,
        "lattice": lattice,
        "dih_node": dih_node,
        "valence": valence_baked,
        "cf": cf_baked,
        "bvs": bvs_baked,
        "forces": forces,
        "magmom": magmom,
        "dos": dos,
        "a2f": a2f,
        "ph_dos": ph_dos,
        "stress": stress,
        "bond_cnt": bond_cnt,
        "bond_nbr": _arr(bond_nbr, np.int32),
        "bond_fea": _arr(bond_fea, np.float32, NBR_FEA_LEN),
        "bond_jimage": _arr(bond_jimage, np.int16, 3),
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
                     use_poly_edges, use_bond_angles, build_angle_bias, *,
                     use_rich_node_features=False, use_dihedrals=False,
                     use_valence_features=False, use_cf_features=False,
                     use_bvs_features=False, mask_oxidation_feature=False,
                     mask_geometry_features=False, use_poly_node_summary=False):
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
    # Per-slot periodic image of the neighbor (center->neighbor), aligned with nbr_idx.
    # Pad slots stay (0,0,0): they self-loop (nbr_idx defaults to i) so the bond vector
    # is exactly 0 and is masked out by bond_pad in the model.
    nbr_jimage = np.zeros((n_atoms, M, 3), dtype=np.int64)

    b0 = v0 = 0
    for i in range(n_atoms):
        c = int(r["bond_cnt"][i])
        k = min(c, M)
        nbr_fea[i, :k, :NBR_FEA_LEN] = r["bond_fea"][b0:b0 + k]
        nbr_idx[i, :k] = r["bond_nbr"][b0:b0 + k]
        nbr_jimage[i, :k] = r["bond_jimage"][b0:b0 + k]

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
    if mask_oxidation_feature:
        # Electronic-configuration consolidation ablation: zero the formal
        # oxidation column (base col 1) so the CF orbital filling is the ONLY
        # electronic-configuration channel. Width unchanged; the zero column
        # normalizes to 0 (std floor 1.0).
        atom_fea[:, 1] = 0.0
    if mask_geometry_features:
        # Composition-only rung of the no-pretrain ladder: zero the geometry-
        # derived base columns (see GEO_FEATURE_COLS). Width unchanged.
        atom_fea[:, GEO_FEATURE_COLS] = 0.0
    if use_rich_node_features:
        # Concat element features looked up from the stored Z (atom_fea[:, 0]).
        rich = torch.from_numpy(rich_node_features(atom_fea[:, 0].numpy()))
        atom_fea = torch.cat([atom_fea, rich], dim=1)
    if use_valence_features:
        # Order [base | rich | valence | cf | dihedral] — every stats path mirrors
        # this. Prefer the builder-baked per-species block (schema >= v4.3, carried
        # as r["valence"]); legacy graphs/packs fall back to the load-time
        # computation from stored Z (col 0) + oxidation (col 1).
        if r.get("valence") is not None:
            val = torch.from_numpy(np.array(r["valence"], dtype=np.float32, copy=True))
        else:
            val = torch.from_numpy(valence_node_features(atom_fea[:, 0].numpy(),
                                                         atom_fea[:, 1].numpy()))
        atom_fea = torch.cat([atom_fea, val], dim=1)
    if use_cf_features:
        # Baked-only (see CF_FEA_LEN note): a cf-less graph/pack is a hard error,
        # never silently zero — a masked union must not mix real and fake CF.
        if r.get("cf") is None:
            raise ValueError(
                "use_cf_features=True but this graph/pack carries no baked cf block "
                "— rebuild it with builder schema >= v4.3 (crystal_field_aom).")
        cf = torch.from_numpy(np.array(r["cf"], dtype=np.float32, copy=True))
        atom_fea = torch.cat([atom_fea, cf], dim=1)
    if use_bvs_features:
        if r.get("bvs") is None:
            raise ValueError(
                "use_bvs_features=True but this graph/pack carries no baked bvs "
                "block — rebuild it with builder schema >= v4.4 (bond_valence).")
        bvs = torch.from_numpy(np.array(r["bvs"], dtype=np.float32, copy=True))
        atom_fea = torch.cat([atom_fea, bvs], dim=1)
    if use_poly_node_summary:
        atom_fea = torch.cat([atom_fea, torch.from_numpy(poly_node_summary(r))], dim=1)
    if use_dihedrals:
        # Concat the per-atom 4-body torsion summary (zeros on a dihedral-less pack).
        # Order is [base | rich | dihedral] -- the stats paths mirror this exactly.
        dih = torch.from_numpy(np.ascontiguousarray(r["dih_node"], dtype=np.float32))
        atom_fea = torch.cat([atom_fea, dih], dim=1)
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
    nbr_jimage_t = torch.from_numpy(nbr_jimage)
    return (atom_fea, nbr_fea_t, nbr_fea_idx, poly_fea_t, poly_fea_idx, nbr_angle,
            frac_coords, lattice, nbr_jimage_t)


# MPtrj stores stress as the RAW VASP tensor in kBar; the model emits stress as the
# thermodynamic sigma = (1/V) dE/dstrain in eV/A^3 (model.py, no sign flip — unlike forces).
# Convert the target to the model's units: kBar -> eV/A^3 is /1602.1766 (1 eV/A^3 =
# 160.21766 GPa = 1602.1766 kBar), and VASP's sign convention is the NEGATIVE of the
# thermodynamic stress, so flip it — otherwise stress (dE/dstrain) fights forces (dE/dr),
# which share the same energy. Without this, target ~kBar (|.|~1-11) vs model ~eV/A^3
# (~1e-2) made stress unlearnable (flat at the mean MAE from epoch 0).
STRESS_KBAR_TO_EVA3 = -1.0 / 1602.1766208


def _assemble_targets(r, energy=float("nan"), bandgap=float("nan"), dos_per_atom=True,
                      eph_lambda=float("nan"), eph_wlog=float("nan")):
    """Build the (targets, masks) dicts for masked multitask training.

    Per-atom: forces (N,3), magmom (N,1) (from the ragged extraction). Per-structure:
    stress (3,3) (ragged, converted kBar->eV/A^3 via STRESS_KBAR_TO_EVA3) + scalar
    energy/bandgap (from the index). An absent target is a NaN sentinel -> its mask is
    False and its value is zeroed, so a masked loss never propagates NaN. Masks are
    per-STRUCTURE bools (a structure carries a given label for every atom or none of them
    -> per-atom force/magmom losses broadcast the structure mask).
    """
    def _t(arr):
        t = torch.from_numpy(np.array(arr, dtype=np.float32, copy=True))
        present = torch.tensor(bool(torch.isfinite(t).all()))
        return torch.nan_to_num(t, nan=0.0), present

    def _scalar(x):
        t = torch.tensor([float(x)], dtype=torch.float32)
        return torch.nan_to_num(t, nan=0.0), torch.tensor(bool(torch.isfinite(t).all()))

    forces, m_f = _t(r["forces"])          # (N,3)
    magmom, m_m = _t(r["magmom"])          # (N,1)
    stress, m_s = _t(r["stress"])          # (3,3) raw kBar ...
    stress = stress * STRESS_KBAR_TO_EVA3  # ... -> eV/A^3, model units (see constant above)
    dos, m_dos = _t(r["dos"])              # (n_energy,) total DOS on the fixed E_F grid
    if dos_per_atom:
        dos = dos / max(int(r["n_atoms"]), 1)  # -> per-atom (intensive) DOS: removes the
                                               # size confound + matches the segment-MEAN
                                               # head. False -> legacy extensive total DOS
                                               # (segment-SUM head), for the old-window
                                               # ablation cell.
    # Phonon spectra on the shared PHONON grid. alpha^2F is INTENSIVE by
    # construction (a Fermi-surface average) — no per-atom division; phonon DOS
    # is EXTENSIVE (3N modes) — per-atom like the electronic DOS, matching the
    # segment-MEAN heads.
    a2f, m_a2f = _t(r["a2f"])              # (PHONON_N_BINS,)
    ph_dos, m_ph = _t(r["ph_dos"])         # (PHONON_N_BINS,)
    ph_dos = ph_dos / max(int(r["n_atoms"]), 1)
    energy_t, m_e = _scalar(energy)        # (1,)
    bandgap_t, m_bg = _scalar(bandgap)     # (1,)
    # Electron-phonon scalars (Cerqueira DFPT set): coupling constant lambda and
    # log-moment omega_log (K, raw — std-normalized by compute_target_stats).
    eph_la_t, m_la = _scalar(eph_lambda)   # (1,)
    eph_wl_t, m_wl = _scalar(eph_wlog)     # (1,)
    targets = {"forces": forces, "magmom": magmom, "stress": stress, "dos": dos,
               "energy": energy_t, "bandgap": bandgap_t,
               "eph_lambda": eph_la_t, "eph_wlog": eph_wl_t,
               "eph_a2f": a2f, "ph_dos": ph_dos}
    masks = {"forces": m_f, "magmom": m_m, "stress": m_s, "dos": m_dos,
             "energy": m_e, "bandgap": m_bg,
             "eph_lambda": m_la, "eph_wlog": m_wl,
             "eph_a2f": m_a2f, "ph_dos": m_ph}
    return targets, masks


def _build_sample(data_row, max_num_nbr, max_num_poly_nbr,
                  use_poly_edges, use_bond_angles, build_angle_bias=False, *,
                  use_rich_node_features=False, use_valence_features=False,
                  use_cf_features=False, use_bvs_features=False,
                  use_dihedrals=False, multitask=False, bandgap=float("nan"),
                  dos_per_atom=True, mask_oxidation_feature=False,
                  mask_geometry_features=False, use_poly_node_summary=False,
                  eph_lambda=float("nan"), eph_wlog=float("nan")):
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
    # Feature flags are KEYWORD-ONLY on both assembly functions: they historically
    # declared use_dihedrals/use_valence_features in OPPOSITE positional order — a
    # transposed positional call is type- and shape-silent (both bools) and would
    # train on the wrong features with no error.
    sample = _assemble_sample(ragged, max_num_nbr, max_num_poly_nbr,
                              use_poly_edges, use_bond_angles, build_angle_bias,
                              use_rich_node_features=use_rich_node_features,
                              use_dihedrals=use_dihedrals,
                              use_valence_features=use_valence_features,
                              use_cf_features=use_cf_features,
                              use_bvs_features=use_bvs_features,
                              mask_oxidation_feature=mask_oxidation_feature,
                              mask_geometry_features=mask_geometry_features,
                              use_poly_node_summary=use_poly_node_summary)

    if multitask:
        targets, masks = _assemble_targets(ragged, energy=target, bandgap=bandgap,
                                           dos_per_atom=dos_per_atom,
                                           eph_lambda=eph_lambda, eph_wlog=eph_wlog)
        return (sample, targets, masks, cif_id)
    target = torch.FloatTensor([float(target)])
    label = torch.LongTensor([int(label)])
    return (sample, target, label, cif_id)


def _sample_to_device(sample, device):
    """Move a built sample's tensors onto ``device`` (the cif_id string is left as-is).
    Arity-agnostic over the input tuple; handles BOTH the single-target sample (target,
    label are tensors) and the multitask sample (they are targets/masks dicts)."""
    sample_in, target, label, cif_id = sample

    def mv(x):
        if isinstance(x, dict):
            return {k: v.to(device) for k, v in x.items()}
        return x.to(device)
    return (
        tuple(t.to(device) for t in sample_in),
        mv(target), mv(label), cif_id,
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
                 build_angle_bias: bool = False, use_rich_node_features: bool = False,
                 use_dihedrals: bool = False, multitask: bool = False,
                 frame_subsample: int = 1, use_valence_features: bool = False,
                 use_cf_features: bool = False, use_bvs_features: bool = False,
                 n_energy: int = None, dos_per_atom: bool = True,
                 mask_oxidation_feature: bool = False,
                 mask_geometry_features: bool = False,
                 use_poly_node_summary: bool = False):
        assert os.path.exists(index_path), f"Index file not found: {index_path}"
        self.use_rich_node_features = use_rich_node_features
        self.use_valence_features = use_valence_features
        self.use_cf_features = use_cf_features
        self.use_bvs_features = use_bvs_features
        self.mask_oxidation_feature = mask_oxidation_feature
        self.mask_geometry_features = mask_geometry_features
        self.use_poly_node_summary = use_poly_node_summary
        self.use_dihedrals = use_dihedrals
        # DOS target width: graph JSONs carry the CURRENT fetch grid only (DOS_N_ENERGY),
        # so this backend can't serve a different width — fail loudly, don't mis-shape.
        # (Old-grid DOS lives only in old PACKS; use the packed backend for those.)
        if multitask and n_energy not in (None, DOS_N_ENERGY):
            raise ValueError(f"CIFDataV4 (graph backend) serves DOS on the current "
                             f"{DOS_N_ENERGY}-bin grid; requested n_energy={n_energy}.")
        self.dos_per_atom = dos_per_atom
        # When True, __getitem__ returns (input, targets_dict, masks_dict, cif_id) for
        # the masked multitask trainer instead of (input, target_scalar, label, cif_id).
        self.multitask = multitask
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
        # Multitask union members may legitimately lack the configured scalar
        # target (the DOS pack has no formation energy) -> tolerate its absence
        # and keep all rows (the missing scalar is masked, other targets train).
        self.target_column = target_key = _select_target_key(
            index_df, target_column, index_path, required=not multitask)
        # Per-id bandgap lookup (multitask target; order-independent so it survives
        # build_data_rows' shuffle/drop). Absent column -> NaN -> masked off. Coerce
        # non-numeric cells (e.g. '' for missing, a repo convention) to NaN.
        self._bandgap_by_id = (
            dict(zip(index_df["id"].astype(str),
                     pd.to_numeric(index_df["bandgap"], errors="coerce")))
            if "bandgap" in index_df.columns else {})
        # Electron-phonon scalar targets (Cerqueira pack indexes only).
        self._eph_lambda_by_id = (
            dict(zip(index_df["id"].astype(str),
                     pd.to_numeric(index_df["eph_lambda"], errors="coerce")))
            if "eph_lambda" in index_df.columns else {})
        self._eph_wlog_by_id = (
            dict(zip(index_df["id"].astype(str),
                     pd.to_numeric(index_df["eph_wlog"], errors="coerce")))
            if "eph_wlog" in index_df.columns else {})

        # `label` (1 = SC, 0 = non-SC) defaults to 1 for older indexes without the
        # column, leaving the regression path unaffected. Rows whose chosen target
        # is missing (NaN — e.g. a source that didn't carry this target) are
        # dropped so they can't poison training.
        # Shared row construction (see build_data_rows — identical for both
        # backends so seeds map to identical splits): element 2 of each data
        # tuple is this backend's graph path; `groups` (parallel mp_id list)
        # enables material-level splits for trajectory datasets.
        self.data, self.groups, self.labels, dropped = build_data_rows(
            index_df, target_key, index_df["graph_path"].tolist(), random_seed,
            keep_all=multitask)
        if dropped:
            print(f"CIFDataV4: dropped {dropped} rows with no '{target_key}' value")
        if frame_subsample and frame_subsample > 1:
            n0 = len(self.data)
            self.data, self.groups, self.labels = subsample_frames_by_group(
                self.data, self.groups, self.labels, frame_subsample)
            print(f"CIFDataV4: frame_subsample={frame_subsample} kept "
                  f"{len(self.data)}/{n0} frames (1-in-{frame_subsample} per material)")

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
                                   self.build_angle_bias,
                                   use_rich_node_features=self.use_rich_node_features,
                                   use_valence_features=self.use_valence_features,
                                   use_cf_features=self.use_cf_features,
                                   use_bvs_features=self.use_bvs_features,
                                   mask_oxidation_feature=getattr(self, 'mask_oxidation_feature', False),
                                   mask_geometry_features=getattr(self, 'mask_geometry_features', False),
                                   use_poly_node_summary=getattr(self, 'use_poly_node_summary', False),
                                   use_dihedrals=self.use_dihedrals,
                                   multitask=self.multitask,
                                   bandgap=self._bandgap_by_id.get(str(rec[0]), float("nan")),
                                   eph_lambda=getattr(self, '_eph_lambda_by_id', {}).get(str(rec[0]), float("nan")),
                                   eph_wlog=getattr(self, '_eph_wlog_by_id', {}).get(str(rec[0]), float("nan")),
                                   dos_per_atom=self.dos_per_atom)
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
                               self.use_bond_angles, self.build_angle_bias,
                               use_rich_node_features=self.use_rich_node_features,
                               use_valence_features=self.use_valence_features,
                               use_cf_features=self.use_cf_features,
                               use_bvs_features=self.use_bvs_features,
                               mask_oxidation_feature=getattr(self, 'mask_oxidation_feature', False),
                               mask_geometry_features=getattr(self, 'mask_geometry_features', False),
                               use_poly_node_summary=getattr(self, 'use_poly_node_summary', False),
                               use_dihedrals=self.use_dihedrals,
                               multitask=self.multitask,
                               bandgap=self._bandgap_by_id.get(str(self.data[idx][0]), float("nan")),
                               eph_lambda=getattr(self, '_eph_lambda_by_id', {}).get(str(self.data[idx][0]), float("nan")),
                               eph_wlog=getattr(self, '_eph_wlog_by_id', {}).get(str(self.data[idx][0]), float("nan")),
                               dos_per_atom=self.dos_per_atom)

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


class ConcatMTDataset:
    """A masked-union of multiple multitask packs (e.g. packed_v4 [MPtrj: E/F/stress/magmom/
    bandgap] + the DOS pack [relaxed MP: dos]) for multitask pretraining. Each pack supplies
    the targets it has; the others are NaN-masked, so the model trains every head over the
    union. Concatenates .data/.labels/.groups so the shared splitter (get_sc_nonsc_loaders,
    resolve_split_by) works unchanged; __getitem__ delegates to the pack owning the global
    index. feature_stats delegates to the first pack (all packs share the cgv4 feature space,
    so they MUST be built with the same feature flags). size_grouped_batches works on the union
    (`dataset_atom_counts` concatenates the members' counts) and is RECOMMENDED: it bounds the
    per-batch atom count (and thus the O(N·M²) local-attention + per-atom-head memory), which is
    what plain batch_size batching leaves uncapped — the rung-04 OOM."""

    def __init__(self, datasets):
        assert datasets, "ConcatMTDataset needs >= 1 dataset"
        assert all(getattr(d, "multitask", False) for d in datasets), \
            "ConcatMTDataset members must be opened with multitask=True"
        self.datasets = list(datasets)
        self._offsets = np.cumsum([0] + [len(d) for d in self.datasets])
        self.multitask = True
        self.is_packed = all(getattr(d, "is_packed", False) for d in self.datasets)
        self.target_column = self.datasets[0].target_column
        self.data = [rec for d in self.datasets for rec in d.data]
        self.labels = [lab for d in self.datasets for lab in d.labels]
        self.groups = ([g for d in self.datasets for g in d.groups]
                       if all(getattr(d, "groups", None) is not None for d in self.datasets)
                       else None)

    def __len__(self):
        return int(self._offsets[-1])

    def __getitem__(self, i):
        d = int(np.searchsorted(self._offsets, i, side="right") - 1)
        return self.datasets[d][i - int(self._offsets[d])]

    def feature_stats(self, indices, max_graphs=4000, seed=123):
        # Respect the passed TRAIN indices (no val/test leakage, unlike using all of
        # pack 0): keep the global-union indices that fall in pack 0 and map them to
        # pack-0-local indices. Packs share the cgv4 feature space (same feature
        # flags), so the dominant first pack is a representative, leakage-free sample.
        d0 = self.datasets[0]
        n0 = len(d0)
        local0 = [i for i in indices if 0 <= i < n0]
        if not local0:                      # no pack-0 rows in this split (pathological)
            local0 = list(range(n0))
        return d0.feature_stats(local0, max_graphs=max_graphs, seed=seed)


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


def load_cif_dataset_from_args(index_path, args, **overrides):
    """Open a dataset with the feature flags recorded in a checkpoint's ``args`` (or a
    training config) — the SINGLE enumeration of the feature-flag list, so every
    consumer (embed_index / embed_raw / FineTune / smoke scripts) builds samples in
    the same feature space as the encoder was trained with. A flag added here reaches
    all of them at once; the per-consumer copies this replaces drifted (e.g.
    use_valence_features had to be retro-patched into embed_index). ``overrides``
    pass straight through to load_cif_dataset (e.g. multitask=True, n_energy=...)."""
    kw = dict(
        target_column=None, build_angle_bias=True,
        max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        use_poly_edges=args.get("use_poly_edges", True),
        use_bond_angles=args.get("use_bond_angles", False),
        use_rich_node_features=args.get("use_rich_node_features", False),
        use_valence_features=args.get("use_valence_features", False),
        use_cf_features=args.get("use_cf_features", False),
        use_bvs_features=args.get("use_bvs_features", False),
        use_dihedrals=args.get("use_dihedrals", False),
        mask_oxidation_feature=args.get("mask_oxidation_feature", False),
        mask_geometry_features=args.get("mask_geometry_features", False),
        use_poly_node_summary=args.get("use_poly_node_summary", False),
    )
    kw.update(overrides)
    return load_cif_dataset(index_path, **kw)


def dataset_ids(dataset):
    """The dataset's cif ids in POSITION order. Rows are seed-shuffled at load
    (build_data_rows), so index-pickle order NEVER matches dataset positions — join
    external tables through these ids (or positions_for_ids), never by index order.
    (Mapping split labels through index order scrambled every FineTune fold's actual
    membership until 2026-07-14.) Works on CIFDataV4/PackedCIFDataV4/ConcatMTDataset."""
    return [rec[0] for rec in dataset.data]


def positions_for_ids(dataset, ids, strict=True):
    """Dataset POSITIONS for ``ids`` (order-preserving) — the safe join between an
    external, index-ordered table and this dataset's rows. ``strict`` raises if any
    id doesn't resolve (a silent drop here is how fold membership goes wrong)."""
    pos = {rec[0]: i for i, rec in enumerate(dataset.data)}
    missing = [i for i in ids if i not in pos]
    if strict and missing:
        raise KeyError(f"{len(missing)} id(s) not in dataset (e.g. {missing[:3]}); "
                       "external table and dataset disagree — refusing a silent drop.")
    return [pos[i] for i in ids if i in pos]


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
    jimages = [sample[8] for (sample, *_rest) in dataset_list]  # each (n_i, M, 3)
    frac_coords = torch.cat(fracs, dim=0)                       # (N, 3)
    lattice = torch.stack(lats, dim=0)                          # (B, 3, 3)
    nbr_jimage = torch.cat(jimages, dim=0)                      # (N, M, 3), aligned w/ nbr_fea_idx
    return base_input + (frac_coords, lattice, nbr_jimage), targets, labels, cif_ids


def collate_pool_multitask(dataset_list):
    """Collate for masked multitask training. Samples are (input_tuple, targets, masks,
    cif_id). Reuses collate_pool_geom for the input batching (placeholder scalar targets),
    then batches the dicts: per-atom targets (forces N,3 / magmom N,1) concat over atoms
    in the SAME order as the batched atom features; per-structure (stress B,3,3 / energy B
    / bandgap B) and ALL masks stack to (B,)."""
    placeholder = [(s[0], torch.zeros(1), torch.zeros(1, dtype=torch.long), s[3])
                   for s in dataset_list]
    base_input, _t, _l, cif_ids = collate_pool_geom(placeholder)
    tds = [s[1] for s in dataset_list]
    mds = [s[2] for s in dataset_list]
    targets = {
        "forces": torch.cat([t["forces"] for t in tds], dim=0),    # (N, 3)
        "magmom": torch.cat([t["magmom"] for t in tds], dim=0),    # (N, 1)
        "stress": torch.stack([t["stress"] for t in tds], dim=0),  # (B, 3, 3)
        "dos": torch.stack([t["dos"] for t in tds], dim=0),        # (B, DOS_N_ENERGY)
        "energy": torch.cat([t["energy"] for t in tds], dim=0),    # (B,)
        "bandgap": torch.cat([t["bandgap"] for t in tds], dim=0),  # (B,)
        "eph_lambda": torch.cat([t["eph_lambda"] for t in tds], dim=0),  # (B,)
        "eph_wlog": torch.cat([t["eph_wlog"] for t in tds], dim=0),      # (B,)
        "eph_a2f": torch.stack([t["eph_a2f"] for t in tds], dim=0),      # (B, PHONON_N_BINS)
        "ph_dos": torch.stack([t["ph_dos"] for t in tds], dim=0),        # (B, PHONON_N_BINS)
    }
    masks = {k: torch.stack([m[k] for m in mds], dim=0) for k in mds[0]}  # each (B,)
    return base_input, targets, masks, cif_ids


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

    ConcatMTDataset (the masked union): concatenate the members' counts in member
    order — exactly the order ``ConcatMTDataset.data`` (and thus the global index)
    uses — so size-grouped batching bounds the union's per-batch atom count too.
    Any member without cheap counts -> None (fall back to plain batching).
    """
    members = getattr(dataset, "datasets", None)
    if members is not None:                       # ConcatMTDataset (a union of packs)
        per = [dataset_atom_counts(d) for d in members]
        return np.concatenate(per) if all(p is not None for p in per) else None
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

    def set_epoch(self, epoch):
        """Advance to a new epoch: drop any cached layout (so the next len()/iter rebuilds
        from the freshly-reshuffled sampler) and forward the epoch to a distributed shard
        sampler underneath. The cached layout MUST be dropped explicitly here because the
        data-parallel trainer breaks each epoch early at the synced min step count, so the
        post-pass auto-clear in __iter__ never runs — without this, the next epoch would
        reuse this epoch's stale (un-reshuffled, mis-sharded) batches."""
        self._pending = None
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

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


class DistributedShardSampler(Sampler):
    """A disjoint per-rank shard of ``indices``, reshuffled each epoch, for the
    explicit-allreduce data-parallel trainer (see common/dist_utils). Deterministically
    shuffles the FULL index list with ``seed + epoch`` (identical on every rank), pads
    to a multiple of ``world_size``, then strides by ``rank`` — the standard
    DistributedSampler scheme, but over our explicit train-index list so it composes
    UNDER ``SizeGroupedBatchSampler`` (which size-groups within the rank's shard).

    ``set_epoch`` MUST be called once per epoch (the trainer does this): all ranks use
    the same epoch -> the same shuffle -> disjoint strided shards that together cover
    the train set. Sharding by index is leakage-free because the material split already
    happened upstream (each shard is a subset of the same train materials)."""

    def __init__(self, indices, world_size, rank, seed=123):
        self.indices = list(indices)
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        order = self.indices[:]
        random.Random(self.seed + self.epoch).shuffle(order)
        pad = (-len(order)) % self.world_size      # make it divisible so shards are equal
        if pad:
            order += order[:pad]
        yield from order[self.rank::self.world_size]

    def __len__(self):
        return (len(self.indices) + self.world_size - 1) // self.world_size


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
                         collate_fn=collate_pool, dist_info=None,
                         train_index_weights=None):
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

    ``train_index_weights`` (per-global-index integer multiplicity, e.g. from a
    masked-union's member_weights) oversamples the TRAIN loader only: train indices
    are replicated w times at sampler construction. Val/test stay at natural
    multiplicity, so early stopping and reported metrics are the uniform per-sample
    statistics, comparable across weighted and unweighted runs; the returned
    train_sc_idx / train_nonsc_idx also stay unweighted (they feed the feature/
    target normalizers, which should reflect the natural data distribution).

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

    # Data-parallel: the TRAIN loader draws from a disjoint per-rank shard (reshuffled
    # each epoch via set_epoch) so the N replicas see different data and together cover
    # the train set; val/test stay whole (rank 0 validates). Non-distributed -> the
    # BalancedEpochSampler path, byte-identical. DDP here targets the all-SC multitask
    # pretraining (ratio=inf -> ns_train unused), so sharding sc_train is the full train set.
    if train_index_weights is not None:
        if sc_to_nonsc_ratio != float("inf"):
            # n_nonsc_train is computed from UNWEIGHTED counts, so weights would
            # silently distort the configured class ratio by the mean SC weight;
            # refuse until the interaction is designed (review 2026-07-28).
            raise NotImplementedError(
                "train_index_weights with a finite sc_to_nonsc_ratio is not "
                "supported (epoch class balance would silently inflate)")
        # Self-enforcing contract (review 2026-07-28): the parallel array must
        # cover the dataset exactly, and fractional/zero weights silently delete
        # samples via int() truncation — refuse both here, not just in gps_main.
        if len(train_index_weights) != len(dataset):
            raise ValueError(f"train_index_weights length {len(train_index_weights)} "
                             f"!= len(dataset) {len(dataset)}")
        _w_arr = np.asarray(train_index_weights)
        if not np.all((_w_arr >= 1) & (_w_arr == np.floor(_w_arr))):
            raise ValueError("train_index_weights must be positive integers "
                             "(fractional/zero would silently drop samples)")

    def _weighted(idx):
        # Train-only oversampling: replicate index i train_index_weights[i] times.
        if train_index_weights is None:
            return idx
        return [i for i in idx for _ in range(int(train_index_weights[i]))]

    sc_train_w, ns_train_w = _weighted(sc_train), _weighted(ns_train)
    train_shard_sampler = None
    if dist_info is not None and getattr(dist_info, "enabled", False):
        train_shard_sampler = DistributedShardSampler(
            sc_train_w, dist_info.world_size, dist_info.rank, seed=seed)
        train_loader = make_loader(train_shard_sampler)
    else:
        train_loader = make_loader(
            BalancedEpochSampler(sc_train_w, ns_train_w, n_nonsc_train, seed=seed))

    return {
        "train": train_loader,
        # The shard sampler (or None) so the trainer can set_epoch it each epoch.
        "train_shard_sampler": train_shard_sampler,
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
            **({"train_sc_weighted": len(sc_train_w)}
               if train_index_weights is not None else {}),
        },
    }
