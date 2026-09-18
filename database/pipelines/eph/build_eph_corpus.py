"""Cerqueira electron-phonon corpus -> cgv4 graphs + index (task #4).

Streams the 8,253 structures of the Cerqueira/Sanna/Marques DFPT set
(database/datafiles/EPH_Cerqueira/DS-{A,B}.pk.bz2: mat_id, pymatgen structure
dict, Tc/la/wlog/dosef/debye) through generate_CGv4_DB_from_structures. The
index carries eph_lambda / eph_wlog as target columns (KNOWN_TARGET_COLUMNS);
there is NO energy column — in the masked union these rows supervise the two
e-ph heads only. Metals: the CF anion gate / BVS electronegativity gate emit
zero blocks on metal-metal sites by construction (intended).

  python database/pipelines/eph/build_eph_corpus.py
  python main.py pack-dataset --index database/datafiles/EPH_Cerqueira/EPH_index.pickle \
      --out database/datafiles/EPH_Cerqueira/eph_pack_v45
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root from database/pipelines/<group>/
sys.path.insert(0, _ROOT)

import pandas as pd  # noqa: E402

from database.database_main import generate_CGv4_DB_from_structures  # noqa: E402

EPH_DIR = os.path.join("database", "datafiles", "EPH_Cerqueira")


def records():
    seen = set()
    for name in ("DS-A", "DS-B"):
        df = pd.read_pickle(os.path.join(EPH_DIR, f"{name}.pk.bz2"))
        for r in df.itertuples(index=False):
            if r.mat_id in seen:
                continue
            seen.add(r.mat_id)
            yield {"id": str(r.mat_id), "structure": r.structure,
                   "mp_id": str(r.mat_id),
                   "eph_lambda": float(r.la), "eph_wlog": float(r.wlog)}


if __name__ == "__main__":
    generate_CGv4_DB_from_structures(
        records(),
        output_dir=os.path.join(EPH_DIR, "graphs_v45_eph"),
        output_index=os.path.join(EPH_DIR, "EPH_index"),
        label=1)
