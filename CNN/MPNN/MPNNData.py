from __future__ import print_function, division

import json
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, Sampler

torch.manual_seed(0)

# --- Feature layout constants ---
# Node features (11 total), in order:
#   Z, oxidation_state, ion_role, chi_pauling, chi_allen,
#   ecn_value, shannon_radius, cn_core,
#   hist_corner, hist_edge, hist_face, hist_other  (← 4 sharing histogram values)
NODE_FEA_LEN = 12

# Edge features (8 total), in order:
#   bond_length, bond_length_over_sum_radii,
#   voronoi_weight_src, voronoi_weight_tgt,
#   ecn_weight_src, ecn_weight_tgt,
#   delta_chi_pauling, coord_sphere
NBR_FEA_LEN = 8

# Index of ecn_weight_src inside the edge feature vector (used for weighted aggregation)
ECN_WEIGHT_SRC_IDX = 4


def _node_to_fea(node: dict) -> np.ndarray:
    return np.array([
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
    ], dtype=np.float32)


def _edge_to_fea(edge: dict) -> np.ndarray:
    return np.array([
        edge["bond_length"],
        edge["bond_length_over_sum_radii"],
        edge["voronoi_weight_src"],
        edge["voronoi_weight_tgt"],
        edge["ecn_weight_src"],
        edge["ecn_weight_tgt"],
        edge["delta_chi_pauling"],
        edge["coord_sphere"],
    ], dtype=np.float32)


class CIFDataV4(Dataset):
    """Dataset that loads pre-computed crystal_graph_v4 JSON files.

    Parameters
    ----------
    index_path : str
        Path to the index pickle produced by generate_CGv4_DB.
    max_num_nbr : int
        Neighbor list will be padded/truncated to this length.
    random_seed : int
    """

    def __init__(self, index_path: str, max_num_nbr: int = 14, random_seed: int = 123):
        assert os.path.exists(index_path), f"Index file not found: {index_path}"

        index_df = pd.read_pickle(index_path)
        # `label` (1 = SC, 0 = non-SC) defaults to 1 for older indexes without the
        # column, leaving the regression path unaffected.
        self.data = [
            (row["id"], row["value"], row["graph_path"],
             int(row["label"]) if "label" in row else 1)
            for _, row in index_df.iterrows()
        ]

        random.seed(random_seed)
        random.shuffle(self.data)
        # labels aligned with __getitem__ index order (after the shuffle above)
        self.labels = [rec[3] for rec in self.data]

        self.max_num_nbr = max_num_nbr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cif_id, target, graph_path, label = self.data[idx]

        with open(graph_path) as f:
            graph = json.load(f)

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

            n_real = len(feas)
            if n_real == 0:
                warnings.warn(f"{cif_id} atom {atom_i} has no neighbors in graph.")

            # Pad to max_num_nbr
            pad = self.max_num_nbr - n_real
            feas.extend([zero_edge] * pad)
            idxs.extend([atom_i] * pad)  # self-loop for padding

            nbr_fea_list.append(np.stack(feas, axis=0))
            nbr_idx_list.append(idxs)

        atom_fea = torch.FloatTensor(atom_fea)
        nbr_fea = torch.FloatTensor(np.stack(nbr_fea_list, axis=0))   # (N, M, 8)
        nbr_fea_idx = torch.LongTensor(nbr_idx_list)                    # (N, M)
        target = torch.FloatTensor([float(target)])
        label = torch.LongTensor([int(label)])

        return (atom_fea, nbr_fea, nbr_fea_idx), target, label, cif_id


def collate_pool(dataset_list):
    """Collate crystals into a batch, offsetting neighbor indices into the batch atom array."""
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target, batch_label = [], [], []
    batch_cif_ids = []
    base_idx = 0
    for (atom_fea, nbr_fea, nbr_fea_idx), target, label, cif_id in dataset_list:
        n_i = atom_fea.shape[0]
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.LongTensor(np.arange(n_i) + base_idx))
        batch_target.append(target)
        batch_label.append(label)
        batch_cif_ids.append(cif_id)
        base_idx += n_i

    return (
        torch.cat(batch_atom_fea, dim=0),
        torch.cat(batch_nbr_fea, dim=0),
        torch.cat(batch_nbr_fea_idx, dim=0),
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

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=pin_memory)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers, collate_fn=collate_fn,
                            pin_memory=pin_memory)

    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler,
                                 num_workers=num_workers, collate_fn=collate_fn,
                                 pin_memory=pin_memory)
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


def get_classification_loaders(dataset, batch_size=64, val_ratio=0.1,
                               test_ratio=0.1, n_nonsc=5000, num_workers=0,
                               pin_memory=False, seed=123):
    """Build SC/non-SC classifier loaders (stratified split, per-epoch balanced
    sampler, realistic + balanced val/test). See OriginalCGCNN.data for details.

    Returns a dict: train, val_realistic, val_balanced, test_realistic,
    test_balanced, split_sizes.
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
        "split_sizes": {
            "train_sc": len(sc_train), "train_nonsc": len(ns_train),
            "val_sc": len(sc_val), "val_nonsc": len(ns_val),
            "test_sc": len(sc_test), "test_nonsc": len(ns_test),
        },
    }
