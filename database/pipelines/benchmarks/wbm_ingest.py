"""Matbench Discovery, energy-only track: WBM test-set ingestion.

Track: predict formation energy on WBM's DFT-RELAXED structures (RS2RE-style
direct prediction — no relaxation by us), score discovery metrics against the
summary's e_form/e_above_hull. Training set: MPtrj energy-only (rung 50) —
the benchmark's sanctioned training data, so no leakage caveats.

Steps (resumable):
  1. stream wbm-cse.jsonl.gz (pymatgen ComputedStructureEntry per line;
     entry.structure = relaxed) -> generate_CGv4_DB_from_structures
  2. write a single test index (id = wbm id, value = e_form target from the
     summary for bookkeeping; predictions come from the model)
  3. pack for fast inference.

Usage: python database/pipelines/benchmarks/wbm_ingest.py [--limit N]
"""
import argparse
import gzip
import json
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root from database/pipelines/<group>/
sys.path.insert(0, _ROOT)
from database.database_main import generate_CGv4_DB_from_structures  # noqa: E402

WBM = "database/datafiles/WBM"
GRAPHS = os.path.join(WBM, "graphs_v45_wbm")
INDEX = os.path.join(WBM, "WBM_eform_index")


def records(limit=None):
    n = 0
    with gzip.open(os.path.join(WBM, "wbm-cse.jsonl.gz"), "rt") as f:
        for line in f:
            row = json.loads(line)
            # jsonl rows: {"material_id": ..., "computed_structure_entry": {...}}
            # (older dumps put the CSE dict at top level — handle both)
            cse = row.get("computed_structure_entry", row)
            wid = row.get("material_id") or cse.get("entry_id") or cse.get("data", {}).get("material_id")
            yield {"id": str(wid), "structure": cse["structure"]}
            n += 1
            if limit and n >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    generate_CGv4_DB_from_structures(records(a.limit), GRAPHS, INDEX)
    idx = pd.read_pickle(INDEX + ".pickle")
    summ = pd.read_csv(os.path.join(WBM, "wbm-summary.csv.gz"))
    id_col = next(c for c in summ.columns if "material_id" in c or c == "id")
    ef_col = next(c for c in summ.columns
                  if "e_form_per_atom_mp2020" in c or c == "e_form_per_atom")
    ef = dict(zip(summ[id_col].astype(str), summ[ef_col]))
    idx["value"] = idx["id"].map(ef)
    idx["label"] = 1
    missing = int(idx["value"].isna().sum())
    idx = idx.dropna(subset=["value"])
    idx[["id", "value", "graph_path", "label"]].to_pickle(
        os.path.join(WBM, "wbm_test_index.pickle"))
    print(f"test index: {len(idx)} rows ({missing} without summary targets); "
          f"pack with: python main.py pack-dataset --index "
          f"{os.path.join(WBM, 'wbm_test_index.pickle')} --out "
          f"{os.path.join(WBM, 'wbm_pack_v45')}")


if __name__ == "__main__":
    main()
