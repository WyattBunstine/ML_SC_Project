"""Data assembly for the T_c head: join embeddings + descriptors + labels + families.

Produces dense matrices (this is tabular-scale work — the whole SC dataset is a
few MB once pooled), with:

- pooled encoder features: [mean over atoms || max over atoms] of the per-atom
  embedding -> (n, 2*D_enc); zero parameters, and the max channel keeps a
  single decisive site visible (the layered-superconductor argument).
- the physical-descriptor bypass vector (descriptors.py).
- 3DSC metadata joined by cif id: sc_class family (rare combo classes folded
  into 'Other'), synth_doped flag. Non-SC rows get family 'non-SC'.

Splits: SC rows are split FIRST, stratified by family (so family-resolved test
metrics have support); non-SC rows are split independently with the same
fractions. The classification stage inherits the SC rows' assignments, so a
T_c test material is never seen by the trunk during classification pretraining.
"""

import os

import numpy as np
import pandas as pd

CORE_FAMILIES = {"Cuprate", "Ferrite", "Heavy_fermion", "Oxide", "Chevrel", "Carbon"}
# Coarse mechanism grouping for the conventional/unconventional differential.
# 'Other' is predominantly electron-phonon; Oxide is ambiguous (bismuthates...)
# and reported separately rather than forced into either group.
FAMILY_GROUP = {
    "Cuprate": "unconventional", "Ferrite": "unconventional",
    "Heavy_fermion": "unconventional",
    "Other": "conventional", "Chevrel": "conventional", "Carbon": "conventional",
    "Oxide": "oxide", "non-SC": "non-SC",
}


def load_family_metadata(metadata_csv: str) -> pd.DataFrame:
    """3DSC metadata keyed by cif filename: family + synth_doped."""
    meta = pd.read_csv(metadata_csv, usecols=["cif", "sc_class", "synth_doped"])
    meta["id"] = meta["cif"].map(os.path.basename)
    meta["family"] = meta["sc_class"].map(
        lambda c: c if c in CORE_FAMILIES else "Other")
    return meta.set_index("id")[["family", "synth_doped"]]


def assemble(index_path: str, embed_dir: str, descriptors_path: str,
             metadata_csv: str, prefix_map: str = "database/MP:database/datafiles/MP",
             pooling: str = "meanmax", aux_meanmax_dir: str = None):
    """Build the head's design matrices. Returns a dict of aligned arrays.

    Rows missing an embedding or a descriptor are dropped (counted); this keeps
    the assembly robust while the background embed pass is still filling in.

    `enc` ([mean||max], n x 2D) is ALWAYS built — it feeds the ridge probe, which
    stays a pooling-independent control. When `pooling` is a learned pool the raw
    per-atom embeddings are also packed CSR-style ('atom_emb' (sum N_i, D) +
    'atom_ptr' (n+1,)) so the head can pool them with trainable parameters; padding
    to max-atoms would be ~93% waste (mean 11 atoms, max 162).

    `aux_meanmax_dir`: a SECOND per-atom embedding dir whose [mean||max] is appended
    to each physical-descriptor row (the complementarity test — e.g. raw mean||max
    folded in beside a learned gps DeepSets pool, to ask whether the encoder adds
    signal orthogonal to composition). Rows missing the aux embedding are dropped."""
    df = pd.read_pickle(index_path)
    desc_blob = pd.read_pickle(descriptors_path)
    desc_table = desc_blob["table"]
    meta = load_family_metadata(metadata_csv)
    need_atoms = pooling != "meanmax"

    ids, enc_rows, phys_rows, tc, label, family, doped = [], [], [], [], [], [], []
    atom_list = []
    missing_embed = missing_desc = missing_aux = 0
    for row in df.itertuples():
        emb_path = os.path.join(embed_dir, row.id + ".npy")
        if not os.path.exists(emb_path):
            missing_embed += 1
            continue
        vec = desc_table.get(row.id)
        if vec is None:
            missing_desc += 1
            continue
        if aux_meanmax_dir is not None:
            aux_path = os.path.join(aux_meanmax_dir, row.id + ".npy")
            if not os.path.exists(aux_path):
                missing_aux += 1
                continue
            aux = np.load(aux_path)
            vec = np.concatenate(
                [np.asarray(vec, dtype=np.float32),
                 aux.mean(0).astype(np.float32), aux.max(0).astype(np.float32)])
        emb = np.load(emb_path)
        enc_rows.append(np.concatenate([emb.mean(0), emb.max(0)]).astype(np.float32))
        if need_atoms:
            atom_list.append(emb.astype(np.float32))
        phys_rows.append(vec)
        ids.append(row.id)
        tc.append(float(row.tc) if not pd.isna(row.tc) else np.nan)
        label.append(int(row.label))
        if row.id in meta.index:
            family.append(meta.loc[row.id, "family"])
            doped.append(bool(meta.loc[row.id, "synth_doped"]))
        else:
            family.append("non-SC" if int(row.label) == 0 else "Other")
            doped.append(False)

    data = {
        "ids": np.asarray(ids),
        "enc": np.stack(enc_rows) if enc_rows else np.zeros((0, 0), np.float32),
        "phys": np.stack(phys_rows) if phys_rows else np.zeros((0, 0), np.float32),
        "tc": np.asarray(tc, dtype=np.float32),
        "label": np.asarray(label, dtype=np.int64),
        "family": np.asarray(family),
        "synth_doped": np.asarray(doped),
        "group": np.asarray([FAMILY_GROUP[f] for f in family]),
        "missing_embed": missing_embed,
        "missing_desc": missing_desc,
    }
    if need_atoms:
        counts = np.array([a.shape[0] for a in atom_list], dtype=np.int64)
        data["atom_emb"] = (np.concatenate(atom_list, axis=0) if atom_list
                            else np.zeros((0, 0), np.float32))
        data["atom_ptr"] = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        data["atom_dim"] = int(atom_list[0].shape[1]) if atom_list else 0
    print(f"head dataset: {len(ids)} rows "
          f"(SC {int((data['label'] == 1).sum())}, non-SC {int((data['label'] == 0).sum())}); "
          f"dropped {missing_embed} missing-embedding, {missing_desc} missing-descriptor"
          + (f", {missing_aux} missing-aux" if aux_meanmax_dir is not None else ""))
    return data


def _parent_id(cif_id: str) -> str:
    """MP parent of a (doped) 3DSC structure, parsed from its filename
    ('...-MP-mp-978986-synth_doped.cif' -> 'mp-978986'). All doped variants of one parent
    share this key. Falls back to the full id (singleton group) when there's no MP tag —
    e.g. the non-SC negatives, which have no doped variants and so split per-row as before."""
    import re
    m = re.search(r"-MP-(mp-\d+)", str(cif_id))
    return m.group(1) if m else str(cif_id)


def make_splits(data, seed: int = 123, val_frac: float = 0.1, test_frac: float = 0.2):
    """Per-row split assignment ('train'/'val'/'test'), SC stratified by family and
    GROUPED BY MP PARENT — every doped variant of a parent lands in the same split, so a
    near-identical doped structure can't leak across train/test (62% of the SC set is
    shared-parent variants). Whole groups are greedily packed into test->val->train per
    family to hit the row-fraction targets. Non-SC negatives have singleton groups, so
    their split is the usual per-row one."""
    import random
    from collections import defaultdict

    n = len(data["ids"])
    split = np.empty(n, dtype=object)
    groups = np.array([_parent_id(i) for i in data["ids"]])

    def grouped_three_way(idx, label):
        g2rows = defaultdict(list)
        for i in idx:
            g2rows[groups[i]].append(int(i))
        # Each group's family (constant across a parent's variants): keeps family
        # stratification while assigning whole parents.
        by_fam = defaultdict(list)
        for g, rows in g2rows.items():
            by_fam[data["family"][rows[0]]].append(g)
        rng = random.Random(seed + label)
        bucket = {"train": [], "val": [], "test": []}
        for fam, gs in by_fam.items():
            rng.shuffle(gs)                                  # deterministic per (seed, label)
            gs.sort(key=lambda g: -len(g2rows[g]))           # largest groups first
            f_rows = sum(len(g2rows[g]) for g in gs)
            target = {"test": test_frac * f_rows, "val": val_frac * f_rows,
                      "train": (1.0 - test_frac - val_frac) * f_rows}
            cur = {"train": 0, "val": 0, "test": 0}
            for g in gs:                                     # each group -> split with most room
                s = max(("test", "val", "train"), key=lambda k: target[k] - cur[k])
                bucket[s] += g2rows[g]
                cur[s] += len(g2rows[g])
        return bucket["train"], bucket["val"], bucket["test"]

    for label in (1, 0):
        idx = np.where(data["label"] == label)[0]
        if not len(idx):
            continue
        tr, va, te = grouped_three_way(idx, label)
        split[tr], split[va], split[te] = "train", "val", "test"
    return split


def family_mae_report(tc_true, tc_pred, families, groups):
    """Per-family and per-group MAE in Kelvin -> nested dict for metrics.json."""
    report = {"overall": _mae_entry(tc_true, tc_pred)}
    for fam in sorted(set(families)):
        m = families == fam
        report[f"family/{fam}"] = _mae_entry(tc_true[m], tc_pred[m])
    for grp in sorted(set(groups)):
        m = groups == grp
        report[f"group/{grp}"] = _mae_entry(tc_true[m], tc_pred[m])
    return report


def _mae_entry(t, p):
    if len(t) == 0:
        return {"n": 0}
    return {
        "n": int(len(t)),
        "mae_K": float(np.abs(t - p).mean()),
        # Scale-free view: families differ ~15x in median T_c, so Kelvin MAE
        # alone misranks them. log1p-space MAE compares relative accuracy.
        "mae_log1pK": float(np.abs(np.log1p(t) - np.log1p(np.maximum(p, 0.0))).mean()),
        "median_tc_K": float(np.median(t)),
        "mean_tc_K": float(t.mean()),
    }
