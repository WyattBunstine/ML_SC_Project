"""Matbench matbench_mp_e_form ingestion: graphs + pack + official 5-fold indices.

Clean-protocol entry (trained from scratch on fold data only — MPtrj pretraining
would leak the benchmark's own relaxed structures/energies via trajectory
endpoints). Data: the hosted json.gz (132,752 relaxed MP structures, e_form
eV/atom). Folds: the matbench package's documented recipe — sklearn
KFold(n_splits=5, shuffle=True, random_state=18012019) over the canonical row
order (the package itself won't install on py3.12; recipe reproduced here).

Steps (resumable — the builder skips existing graphs):
  1. stream rows -> generate_CGv4_DB_from_structures -> graphs_v45_eform/
  2. pack-dataset -> eform_pack_v45 (train speed)
  3. per fold f: mb_f<k>_train.pickle / mb_f<k>_test.pickle (id, value=e_form,
     graph_path, label) — gps_main carves its own val from the train pickle,
     so test rows are never seen before the parity pass.

Usage: python database/pipelines/benchmarks/matbench_ingest.py
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root from database/pipelines/<group>/
sys.path.insert(0, _ROOT)
from database.database_main import generate_CGv4_DB_from_structures  # noqa: E402

MB = "database/datafiles/Matbench"
GRAPHS = os.path.join(MB, "graphs_v45_eform")
INDEX = os.path.join(MB, "MB_eform_index")
KFOLD_SEED = 18012019   # matbench's fixed fold seed (Dunn et al. 2020 protocol)


def records():
    with gzip.open(os.path.join(MB, "matbench_mp_e_form.json.gz")) as f:
        raw = json.load(f)
    for i, (struct, ef) in enumerate(raw["data"]):
        yield {"id": f"mb-{i:06d}", "structure": struct,
               "formation_energy_per_atom": float(ef)}


def main():
    if not os.path.exists(INDEX + ".pickle"):
        generate_CGv4_DB_from_structures(records(), GRAPHS, INDEX)
    idx = pd.read_pickle(INDEX + ".pickle")
    print(f"index: {len(idx)} rows; cols {list(idx.columns)}")
    # canonical order = the mb-<i> ingestion order == the dataset's row order
    idx = idx.sort_values("id").reset_index(drop=True)
    idx["value"] = idx["formation_energy_per_atom"]
    idx["label"] = 1
    keep = ["id", "value", "graph_path", "label"]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=KFOLD_SEED)
    for k, (tr, te) in enumerate(kf.split(idx)):
        idx.iloc[tr][keep].to_pickle(os.path.join(MB, f"mb_f{k}_train.pickle"))
        idx.iloc[te][keep].to_pickle(os.path.join(MB, f"mb_f{k}_test.pickle"))
        print(f"fold {k}: train {len(tr)} / test {len(te)}")
    print("done — pack with: python main.py pack-dataset --index "
          f"{INDEX}.pickle --out {os.path.join(MB, 'eform_pack_v45')}")


if __name__ == "__main__":
    main()
