"""Why the un-built SuperCon candidates failed, and what would unlock them.

Splits the unbuilt candidates into (a) NO PARENT in the MP pool and (b) parent
found but doping failed; classifies each by chemistry; ranks the missing
parents by how many entries a downloaded CIF would unlock (the ICSD request
list, same idea as the 2026-08 O-interstitial ICSD_Parent_Cifs round).
  python scripts/supercon_expansion/diagnose_failures.py [--sample N]
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_HERE))
for _p in (_ROOT, _HERE, _os.path.join(_ROOT, "scripts", "nemad_expansion")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse, heapq, pickle, warnings  # noqa: E402
from collections import defaultdict, Counter  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
warnings.filterwarnings("ignore")
from pymatgen.core import Composition  # noqa: E402
from formula_match import chem_dict, formula_similarity  # noqa: E402
from synth_dope import synth_dope_one  # noqa: E402
from build_supercon_v7 import (CAND, POOL, SOURCE, METALS, alloy_dope_one, multi_sub_dope_one,
                               _subsets, K, SCDB)  # noqa: E402

STRUCT_CACHE = _os.path.join(SCDB, "parent_structs.pkl")


def cls(f):
    try:
        c = set(Composition(f).get_el_amt_dict())
    except Exception:  # noqa: BLE001
        return "?"
    if c <= METALS:
        return "alloy (all-metal)"
    if "Cu" in c and "O" in c:
        return "cuprate"
    if "O" in c:
        return "other oxide"
    if c & {"S", "Se", "Te"}:
        return "chalcogenide"
    if c & {"N", "C", "B", "P", "As", "Si"}:
        return "pnictide/carbide/boride"
    return "other"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", type=int, default=1200); a = ap.parse_args()
    cand = pd.read_csv(CAND)
    built = set(pd.read_csv(SOURCE)["k"]) if _os.path.exists(SOURCE) else set()
    fail = cand[~cand.k.isin(built)].copy()
    fail["cls"] = fail.formula.map(cls)
    print(f"candidates {len(cand)} | built {len(built)} | UNBUILT {len(fail)}")
    pool = pd.read_pickle(POOL); pool = pool[~pool.material_id.str.startswith("__done__")].copy()
    pool["cd"] = pool["reduced"].map(chem_dict)
    by_sys = defaultdict(list)
    for r in pool.itertuples():
        by_sys[frozenset(r.cd)].append((r.material_id, r.cd, r.eah, r.formula_pretty))
    topk = {}
    for c in fail.itertuples():
        cd_sc = chem_dict(c.formula); S = frozenset(cd_sc)
        acc = []
        subsets = [S] if len(S) == 1 else [frozenset(x) for x in _subsets(sorted(S))]
        for sub in subsets:
            for mid, cd2, eah, pf in by_sys.get(sub, ()):
                tier, trd = formula_similarity(cd_sc, cd2)
                if not np.isnan(tier):
                    acc.append((int(tier), round(eah, 4), round(trd, 5), mid, pf))
        if set(S) <= METALS:
            major = max(cd_sc, key=cd_sc.get)
            for e in S:
                for mid, cd2, eah, pf in by_sys.get(frozenset({e}), ()):
                    acc.append((4 if e == major else 5, round(eah, 4), 1.0, mid, pf))
        for sub in subsets:                       # the matcher's fallback tier
            for mid, cd2, eah, pf in by_sys.get(sub, ()):
                acc.append((6 + (len(S) - len(sub)), round(eah, 4), 1.0, mid, pf))
        if acc:
            topk[c.k] = heapq.nsmallest(K, acc)
    fail["has_parent"] = fail.k.isin(topk)
    print("\n=== (A) unbuilt by chemistry and parent availability ===")
    print(pd.crosstab(fail["cls"], fail.has_parent.map({True: "parent found (doping failed)", False: "NO parent in MP pool"})).to_string())
    # ---- (B) doping reasons on a sample of the parent-found bucket ----
    withp = fail[fail.has_parent]
    samp = withp.sample(min(a.sample, len(withp)), random_state=0)
    mids = sorted({t[3] for k in samp.k for t in topk[k]})
    cache = {}
    if _os.path.exists(STRUCT_CACHE):
        cache = pickle.load(open(STRUCT_CACHE, "rb"))
    need = [m for m in mids if m not in cache]
    if need:
        from mp_api.client import MPRester
        with MPRester(_os.environ["MP_API_KEY"]) as mpr:
            for i in range(0, len(need), 400):
                for d in mpr.materials.summary.search(material_ids=need[i:i + 400], fields=["material_id", "structure"]):
                    cache[str(d.material_id)] = d.structure
        pickle.dump(cache, open(STRUCT_CACHE, "wb"))
    reasons, by_cls = Counter(), defaultdict(Counter)
    for c in samp.itertuples():
        best = None
        for tier, eah, trd, mid, pf in topk[c.k]:
            st0 = cache.get(mid)
            if st0 is None:
                best = best or "no structure"; continue
            try:
                st, r = synth_dope_one(st0, c.formula)
            except Exception as e:  # noqa: BLE001
                st, r = None, f"exception:{type(e).__name__}"
            if st is None:
                try:
                    st, r2 = alloy_dope_one(st0, c.formula)
                except Exception:  # noqa: BLE001
                    st, r2 = None, "alloy exception"
            if st is None:
                try:
                    st, r3 = multi_sub_dope_one(st0, c.formula)
                except Exception:  # noqa: BLE001
                    st, r3 = None, "multisub exception"
                if st is None:
                    best = best or r3
                    continue
            best = "BUILDABLE"; break
        reasons[best] += 1; by_cls[c.cls][best] += 1
    print(f"\n=== (B) doping outcome on {len(samp)} sampled parent-found candidates ===")
    for r, n in reasons.most_common(12):
        print(f"  {n:5d}  {r}")
    print("\n  by chemistry (top reason each):")
    for cl, ctr in sorted(by_cls.items(), key=lambda x: -sum(x[1].values())):
        top = ", ".join(f"{r} {n}" for r, n in ctr.most_common(3))
        print(f"    {cl:26s} n={sum(ctr.values()):4d}  {top}")
    # ---- (C) the ICSD request list: which missing parents unlock the most ----
    nop = fail[~fail.has_parent].copy()
    nop["parent_guess"] = nop.formula.map(lambda f: Composition(f).reduced_formula)
    print(f"\n=== (C) NO-parent candidates: {len(nop)} — ranked by chemical system (a CIF for one unlocks all its rows) ===")
    g = nop.groupby("chemsys").agg(n=("formula", "size"), tc_max=("tc", "max"), tc_med=("tc", "median"),
                                   example=("formula", "first"), cls=("cls", "first"))
    print(g[g.n >= 8].sort_values("n", ascending=False).head(30).round(2).to_string())
    g.sort_values("n", ascending=False).to_csv(_os.path.join(SCDB, "icsd_request_systems.csv"))
    print(f"\n  full ranking -> {_os.path.join(SCDB, 'icsd_request_systems.csv')} ({len(g)} systems)")


if __name__ == "__main__":
    main()
