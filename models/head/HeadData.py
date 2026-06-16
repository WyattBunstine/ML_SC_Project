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
             metadata_csv: str, prefix_map: str = "database/MP:database/datafiles/MP"):
    """Build the head's design matrices. Returns a dict of aligned arrays.

    Rows missing an embedding or a descriptor are dropped (counted); this keeps
    the assembly robust while the background embed pass is still filling in.
    """
    df = pd.read_pickle(index_path)
    desc_blob = pd.read_pickle(descriptors_path)
    desc_table = desc_blob["table"]
    meta = load_family_metadata(metadata_csv)

    ids, enc_rows, phys_rows, tc, label, family, doped = [], [], [], [], [], [], []
    missing_embed = missing_desc = 0
    for row in df.itertuples():
        emb_path = os.path.join(embed_dir, row.id + ".npy")
        if not os.path.exists(emb_path):
            missing_embed += 1
            continue
        vec = desc_table.get(row.id)
        if vec is None:
            missing_desc += 1
            continue
        emb = np.load(emb_path)
        enc_rows.append(np.concatenate([emb.mean(0), emb.max(0)]).astype(np.float32))
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
    print(f"head dataset: {len(ids)} rows "
          f"(SC {int((data['label'] == 1).sum())}, non-SC {int((data['label'] == 0).sum())}); "
          f"dropped {missing_embed} missing-embedding, {missing_desc} missing-descriptor")
    return data


def make_splits(data, seed: int = 123, val_frac: float = 0.1, test_frac: float = 0.2):
    """Per-row split assignment ('train'/'val'/'test'), SC stratified by family."""
    from sklearn.model_selection import train_test_split

    n = len(data["ids"])
    split = np.empty(n, dtype=object)
    sc_idx = np.where(data["label"] == 1)[0]
    nonsc_idx = np.where(data["label"] == 0)[0]

    def three_way(idx, strat):
        trainval, test = train_test_split(
            idx, test_size=test_frac, random_state=seed, stratify=strat)
        strat_tv = None if strat is None else strat[np.isin(idx, trainval)]
        train, val = train_test_split(
            trainval, test_size=val_frac / (1.0 - test_frac),
            random_state=seed, stratify=strat_tv)
        return train, val, test

    sc_train, sc_val, sc_test = three_way(sc_idx, data["family"][sc_idx])
    split[sc_train], split[sc_val], split[sc_test] = "train", "val", "test"
    if len(nonsc_idx):
        ns_train, ns_val, ns_test = three_way(nonsc_idx, None)
        split[ns_train], split[ns_val], split[ns_test] = "train", "val", "test"
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
