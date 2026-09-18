"""Phase-1 matching of magnetic negatives to MP structures — DIRECT matches only.

Tier 1 (identical) / tier 2 (similar) against the magnetic + SC crystal pools;
no synthetic doping in phase 1 (ordered magnetic materials are mostly
stoichiometric; the doped tail can join later via the combined doper). Best
candidate per composition = (tier asc, e_above_hull asc), mirroring the SC
pipeline's tier-first rule.

Output: database/datafiles/NE_SCDB/magnetic_matches.csv
        (key, formula, T_order, order, n_reports, material_id, formula_pretty,
         tier, trd, eah)
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE))))

import re
from collections import defaultdict

import numpy as np
import pandas as pd

from formula_match import chem_dict, formula_similarity


def main():
    neg = pd.read_csv("database/datafiles/NE_SCDB/magnetic_negatives.csv")
    pools = pd.concat([
        pd.read_pickle("database/datafiles/NE_SCDB/mp_crystal_pool_magnetic.pickle"),
        pd.read_pickle("database/datafiles/NE_SCDB/mp_crystal_pool.pickle"),
    ]).drop_duplicates("material_id")
    by_sys = defaultdict(list)
    for r in pools.itertuples():
        cd = chem_dict(r.reduced)
        by_sys[frozenset(cd)].append((r.material_id, cd, r.eah, r.formula_pretty))
    print(f"pool: {len(pools)} materials, {len(by_sys)} systems")

    rows, matched = [], 0
    for r in neg.itertuples():
        cd = {ev.rstrip("0123456789."): float(re.sub(r"^[A-Za-z]+", "", ev))
              for ev in r.key.split("|")}
        best = None
        for mid, pcd, eah, fp in by_sys.get(frozenset(cd), ()):
            tier, trd = formula_similarity(cd, pcd)
            if tier in (1, 2):
                cand = (tier, eah if eah is not None else 9.9, mid, trd, fp)
                if best is None or cand[:2] < best[:2]:
                    best = cand
        if best:
            matched += 1
            rows.append(dict(key=r.key, formula=r.formula, T_order=r.T_order,
                             order=r.order, n_reports=r.n_reports,
                             material_id=best[2], formula_pretty=best[4],
                             tier=best[0], trd=round(best[3], 4), eah=best[1]))
    out = pd.DataFrame(rows)
    print(f"matched {matched}/{len(neg)} "
          f"(tier1 {int((out.tier == 1).sum())}, tier2 {int((out.tier == 2).sum())}); "
          f"unique MP structures: {out.material_id.nunique()}")
    print("order split of matched:", out.order.value_counts().to_dict())
    out.to_csv("database/datafiles/NE_SCDB/magnetic_matches.csv", index=False)
    print("wrote database/datafiles/NE_SCDB/magnetic_matches.csv")


if __name__ == "__main__":
    main()
