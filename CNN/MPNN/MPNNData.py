from __future__ import print_function, division

import json
import os
import random
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

# Edge features (8 total), in order:
#   bond_length, bond_length_over_sum_radii,
#   voronoi_weight_src, voronoi_weight_tgt,
#   ecn_weight_src, ecn_weight_tgt,
#   delta_chi_pauling, coord_sphere
NBR_FEA_LEN = 8

# Index of ecn_weight_src inside the edge feature vector (used for weighted aggregation)
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


def _edge_to_fea(edge: dict) -> np.ndarray:
    # nan_to_num for the same reason as _node_to_fea: delta_chi_pauling is NaN on
    # any edge touching a noble-gas atom (undefined Pauling electronegativity).
    return np.nan_to_num(np.array([
        edge["bond_length"],
        edge["bond_length_over_sum_radii"],
        edge["voronoi_weight_src"],
        edge["voronoi_weight_tgt"],
        edge["ecn_weight_src"],
        edge["ecn_weight_tgt"],
        edge["delta_chi_pauling"],
        edge["coord_sphere"],
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


def compute_feature_stats(dataset, indices, max_graphs=4000, seed=123):
    """Per-feature mean/std for node, bonding-edge, and poly-edge features.

    Computed over (a random sample of) the given *training* indices only — no
    val/test leakage. Stats come from the real graph entries (padding is never
    included). Zero-variance columns get std = 1.0 so normalization just centers
    them instead of dividing by ~0.

    Returns {"node": (mean, std), "edge": (mean, std), "poly": (mean, std)} as
    float32 arrays of length NODE_FEA_LEN / NBR_FEA_LEN / POLY_FEA_LEN.
    """
    idx = list(indices)
    if max_graphs and len(idx) > max_graphs:
        idx = random.Random(seed).sample(idx, max_graphs)

    node_rows, edge_rows, poly_rows = [], [], []
    for i in idx:
        graph = dataset._read_graph(dataset.data[i][2])
        node_rows.extend(_node_to_fea(n) for n in graph["nodes"])
        edge_rows.extend(_edge_to_fea(e) for e in graph["edges"])
        poly_rows.extend(_poly_edge_to_fea(pe) for pe in graph.get("poly_edges", []))

    def _stats(rows, dim):
        if not rows:
            return (np.zeros(dim, np.float32), np.ones(dim, np.float32))
        arr = np.stack(rows).astype(np.float64)
        # nan-safe: a single NaN in a column makes plain mean/std NaN for that
        # column, which then poisons every normalized forward pass. The feature
        # builders already coerce NaN->0, but compute mean/std nan-safely too so a
        # future NaN-bearing feature can't silently produce NaN stats. Columns that
        # are entirely NaN (nanmean/nanstd -> NaN) fall back to mean 0 / std 1.
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
        mean = np.nan_to_num(mean, nan=0.0)
        std = np.nan_to_num(std, nan=1.0)
        std[std < 1e-6] = 1.0
        return (mean.astype(np.float32), std.astype(np.float32))

    return {
        "node": _stats(node_rows, NODE_FEA_LEN),
        "edge": _stats(edge_rows, NBR_FEA_LEN),
        "poly": _stats(poly_rows, POLY_FEA_LEN),
    }


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
                 random_seed: int = 123, target_column: str = None):
        assert os.path.exists(index_path), f"Index file not found: {index_path}"

        index_df = pd.read_pickle(index_path)

        # Pick which column supplies the regression target. A multi-target index
        # (e.g. the MP energy dataset) carries several named target columns;
        # `target_column` (from the config) selects one. With it unset we fall
        # back to the legacy `value` column, then `tc`, so older single-target
        # indexes keep working unchanged. The fallback only accepts a column that
        # actually has data — a tc-less index (energy dataset) has an all-empty
        # `value`, so this raises and tells the user to set `target_column`
        # instead of silently regressing on the wrong target.
        structural_cols = {"id", "value", "graph_path", "label"}
        named_targets = [c for c in index_df.columns
                         if c not in structural_cols and not index_df[c].isna().all()]
        if target_column is not None:
            if target_column not in index_df.columns:
                raise ValueError(
                    f"target_column '{target_column}' not found in index "
                    f"{index_path}; available columns: {list(index_df.columns)}")
            target_key = target_column
        else:
            target_key = next(
                (c for c in ("value", "tc")
                 if c in index_df.columns and not index_df[c].isna().all()),
                None)
            if target_key is None:
                raise ValueError(
                    f"No usable default target ('value'/'tc' absent or all-empty) "
                    f"in index {index_path}. Set 'target_column' in the config to "
                    f"one of {named_targets}.")
        self.target_column = target_key

        # `label` (1 = SC, 0 = non-SC) defaults to 1 for older indexes without the
        # column, leaving the regression path unaffected. Rows whose chosen target
        # is missing (NaN — e.g. a source that didn't carry this target) are
        # dropped so they can't poison training.
        self.data = []
        dropped = 0
        for _, row in index_df.iterrows():
            value = row[target_key]
            if pd.isna(value):
                dropped += 1
                continue
            self.data.append(
                (row["id"], value, row["graph_path"],
                 int(row["label"]) if "label" in row else 1))
        if dropped:
            print(f"CIFDataV4: dropped {dropped} rows with no '{target_key}' value")

        random.seed(random_seed)
        random.shuffle(self.data)
        # labels aligned with __getitem__ index order (after the shuffle above)
        self.labels = [rec[3] for rec in self.data]

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

    def __getitem__(self, idx):
        # Serve the fully-built sample from the per-process LRU cache if present.
        # __getitem__ is deterministic (immutable graphs), so caching the built
        # tensors skips all the neighbor sorting / padding / stacking below — the
        # dominant CPU cost — on every epoch after the first. collate never mutates
        # these tensors, so returning the same objects across epochs is safe.
        cached = self._item_cache.get(idx)
        if cached is not None:
            self._item_cache.move_to_end(idx)   # mark most-recently-used
            return cached

        cif_id, target, graph_path, label = self.data[idx]

        graph = self._read_graph(graph_path)

        nodes = graph["nodes"]
        edges_by_id = {e["id"]: e for e in graph["edges"]}
        adjacency = graph["adjacency"]  # str(node_id) -> [[edge_id, neighbor_id], ...]

        n_atoms = len(nodes)

        # Build atom feature matrix: (n_atoms, NODE_FEA_LEN)
        atom_fea = np.stack([_node_to_fea(n) for n in nodes], axis=0)

        # Build padded neighbor feature and index tensors
        nbr_fea_list = []
        nbr_idx_list = []
        zero_edge = np.zeros(NBR_FEA_LEN, dtype=np.float32)

        for atom_i in range(n_atoms):
            neighbors = adjacency.get(str(atom_i), [])

            # Sort by bond_length so we consistently pick the closest if truncating
            def _bond_len(pair):
                eid = pair[0]
                e = edges_by_id.get(eid)
                return e["bond_length"] if e is not None else 1e9

            neighbors = sorted(neighbors, key=_bond_len)

            feas, idxs = [], []
            for eid, nbr_id in neighbors[:self.max_num_nbr]:
                edge = edges_by_id.get(eid)
                if edge is None:
                    continue
                feas.append(_edge_to_fea(edge))
                idxs.append(nbr_id)

            # An atom with no bonding neighbors (e.g. an isolated noble-gas atom)
            # is benign: the loop below pads it to max_num_nbr with zero edges and
            # self-loops, so it contributes no message. No warning — these are
            # common in the MP dataset and would otherwise spam one line per atom.
            n_real = len(feas)

            # Pad to max_num_nbr
            pad = self.max_num_nbr - n_real
            feas.extend([zero_edge] * pad)
            idxs.extend([atom_i] * pad)  # self-loop for padding

            nbr_fea_list.append(np.stack(feas, axis=0))
            nbr_idx_list.append(idxs)

        # Build padded polyhedral neighbor tensors (second edge type).
        # Older graph JSONs may predate poly edges; default to empty so those
        # crystals still load (poly contribution is then all-padding / zero).
        poly_by_id = {pe["id"]: pe for pe in graph.get("poly_edges", [])}
        poly_adjacency = graph.get("poly_adjacency", {})
        poly_fea_list = []
        poly_idx_list = []
        zero_poly = np.zeros(POLY_FEA_LEN, dtype=np.float32)

        for atom_i in range(n_atoms):
            neighbors = poly_adjacency.get(str(atom_i), [])

            # Sort by direct_distance so we keep the closest if truncating.
            def _poly_dist(pair):
                pe = poly_by_id.get(pair[0])
                return pe["direct_distance"] if pe is not None else 1e9

            neighbors = sorted(neighbors, key=_poly_dist)

            feas, idxs = [], []
            for pid, nbr_id in neighbors[:self.max_num_poly_nbr]:
                pe = poly_by_id.get(pid)
                if pe is None:
                    continue
                feas.append(_poly_edge_to_fea(pe))
                idxs.append(nbr_id)

            # Pad to max_num_poly_nbr (no warning — having no poly edges is
            # normal, e.g. isolated atoms or molecular fragments).
            pad = self.max_num_poly_nbr - len(feas)
            feas.extend([zero_poly] * pad)
            idxs.extend([atom_i] * pad)  # self-loop for padding

            poly_fea_list.append(np.stack(feas, axis=0))
            poly_idx_list.append(idxs)

        atom_fea = torch.FloatTensor(atom_fea)
        nbr_fea = torch.FloatTensor(np.stack(nbr_fea_list, axis=0))   # (N, M, 8)
        nbr_fea_idx = torch.LongTensor(nbr_idx_list)                    # (N, M)
        poly_fea = torch.FloatTensor(np.stack(poly_fea_list, axis=0))  # (N, Mp, 7)
        poly_fea_idx = torch.LongTensor(poly_idx_list)                  # (N, Mp)
        target = torch.FloatTensor([float(target)])
        label = torch.LongTensor([int(label)])

        result = (atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx), target, label, cif_id

        # Store the built sample for reuse on later epochs (LRU-bounded).
        self._item_cache[idx] = result
        if self.graph_cache_size and len(self._item_cache) > self.graph_cache_size:
            self._item_cache.popitem(last=False)   # evict least-recently-used
        return result


def collate_pool(dataset_list):
    """Collate crystals into a batch, offsetting neighbor indices into the batch atom array."""
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    batch_poly_fea, batch_poly_fea_idx = [], []
    crystal_atom_idx, batch_target, batch_label = [], [], []
    batch_cif_ids = []
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx), target, label, cif_id in dataset_list:
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        batch_poly_fea.append(poly_fea)
        batch_poly_fea_idx.append(poly_fea_idx + base_idx)
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        batch_target.append(target)
        batch_label.append(label)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
        torch.cat(batch_poly_fea, dim=0),
        torch.cat(batch_poly_fea_idx, dim=0),
        crystal_atom_idx,
    ), torch.stack(batch_target, dim=0), torch.cat(batch_label, dim=0), batch_cif_ids


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
                         pin_memory=False, seed=123):
    """Build SC/non-SC loaders shared by the regression and classification tasks.

    Stratified per-class split into train/val/test. The train loader draws ALL
    SC-train indices plus a fresh random sample of non-SC each epoch, sized by
    ``sc_to_nonsc_ratio`` (see ``nonsc_count_for_ratio``) and reshuffled every
    epoch via BalancedEpochSampler. val/test are provided in both a "realistic"
    (full class proportions) and a "balanced" (equal SC/non-SC) form.

    With the default ratio of inf, no non-SC are used: the train set is SC-only
    and, when the dataset itself is SC-only, every split collapses to SC-only —
    matching the original behaviour.

    Returns a dict: train, val_realistic, val_balanced, test_realistic,
    test_balanced, split_sizes, and the train-pool index lists
    (train_sc_idx, train_nonsc_idx) + n_nonsc_per_epoch for the caller's use
    (e.g. building the regression target normalizer).
    """
    labels = np.asarray(dataset.labels)
    rng = random.Random(seed)

    def split_class(idx_list):
        idx = list(idx_list)
        rng.shuffle(idx)
        n = len(idx)
        n_test = int(test_ratio * n)
        n_val = int(val_ratio * n)
        return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]

    sc_train, sc_val, sc_test = split_class(np.where(labels == 1)[0].tolist())
    ns_train, ns_val, ns_test = split_class(np.where(labels == 0)[0].tolist())

    n_nonsc = nonsc_count_for_ratio(len(sc_train), sc_to_nonsc_ratio, len(ns_train))

    def make_loader(sampler):
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, collate_fn=collate_pool,
                          pin_memory=pin_memory,
                          persistent_workers=(num_workers > 0))

    def balanced_subset(min_idx, maj_idx):
        k = min(len(min_idx), len(maj_idx))
        return list(min_idx) + rng.sample(maj_idx, k)

    train_loader = make_loader(
        BalancedEpochSampler(sc_train, ns_train, n_nonsc, seed=seed))

    return {
        "train": train_loader,
        "val_realistic": make_loader(SubsetRandomSampler(sc_val + ns_val)),
        "val_balanced": make_loader(SubsetRandomSampler(balanced_subset(sc_val, ns_val))),
        "test_realistic": make_loader(SubsetRandomSampler(sc_test + ns_test)),
        "test_balanced": make_loader(SubsetRandomSampler(balanced_subset(sc_test, ns_test))),
        "train_sc_idx": sc_train,
        "train_nonsc_idx": ns_train,
        "n_nonsc_per_epoch": n_nonsc,
        "split_sizes": {
            "train_sc": len(sc_train), "train_nonsc": len(ns_train),
            "train_nonsc_per_epoch": n_nonsc,
            "val_sc": len(sc_val), "val_nonsc": len(ns_val),
            "test_sc": len(sc_test), "test_nonsc": len(ns_test),
        },
    }
