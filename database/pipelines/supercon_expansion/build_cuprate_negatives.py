"""Curated Cu-oxide Tc=0 negatives (user, 2026-09-11): the counter-evidence the
2026-07 magnetic flood never supplied.

That flood added 5,962 magnetically ordered rows but only ~160 contained Cu+O,
so the cuprate manifold had almost nothing pushing back and both M-arms lost
dome modulation. This builds the NEAR negatives instead: every Cu-bearing
composition in the NEMAD magnetic set (measured Curie/Neel ordering, therefore
a measured non-superconductor) matched to a real structure with the same
machinery as the SuperCon expansion (multi-sublattice / alloy / 3DSC dopers,
ICSD parents, deep subset search).

Overlap policy is family-aware, as settled in 2026-07: a candidate is dropped
only when its composition formula-matches a Tc>0 row (coexistence ambiguity);
undoped SC-family parents are KEPT - they are the highest-value negatives.
Rows carry tc=0 and label=1 (plain regression, no hurdle) plus mag_order for
later use.

  candidates | pool | match-dope | graphs | merge
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
for _p in (_ROOT, _HERE, _os.path.join(_ROOT, "database", "pipelines", "nemad_expansion")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import argparse, re, warnings  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
warnings.filterwarnings("ignore")
from pymatgen.core import Composition  # noqa: E402
import build_supercon_v7 as B  # noqa: E402

MP = B.MP
SCDB = B.SCDB
RAW = _os.path.join(_ROOT, "database", "datafiles", "NE_SCDB", "magnetic_materials.csv")
CAND = _os.path.join(SCDB, "cuneg_candidates.csv")
SYSTEMS = _os.path.join(SCDB, "cuneg_systems.txt")
SOURCE = _os.path.join(MP, "SC_MP_V9_cuneg_source.csv")
CIFS = _os.path.join(MP, "cifs_v9_cuneg")
IDPROP = _os.path.join(SCDB, "id_prop_v9.csv")
GRAPHS = _os.path.join(MP, "graphs_v4_v9_cuneg")
INDEX_OUT = _os.path.join(MP, "SC_MP_V9_cuneg_only")   # negatives-only index; the MERGED V9 is SC_MP_V9_cuneg (was a name collision until 2026-09-18)


def order_of(row):
    for c, lab in (("Neel", "AFM"), ("Neel(Tn)", "AFM"), ("Curie", "FM"), ("Curie(Tc)", "FM")):
        v = row.get(c)
        if isinstance(v, str) and re.search(r"\d", v):
            return lab
        if isinstance(v, (int, float)) and not pd.isna(v):
            return lab
    return None


def cmd_candidates(**_):
    raw = pd.read_csv(RAW, low_memory=False)
    v8 = pd.read_pickle(_os.path.join(MP, "SC_MP_V8_supercon.pickle"))
    v8["f"] = v8.id.str.split("-MP-").str[0].str.split("-ICSD-").str[0]
    v8["k"] = v8.f.map(B.norm_key)
    pos_keys = set(v8[v8.tc > 0].k.dropna())          # family-aware: only Tc>0 blocks a candidate
    have_keys = set(v8.k.dropna())
    rows = []
    for r in raw.to_dict("records"):
        name = str(r.get("Material_Name", "")).strip()
        if "Cu" not in name:
            continue
        try:
            c = Composition(name).get_el_amt_dict()
        except Exception:  # noqa: BLE001
            continue
        if "Cu" not in c or "O" not in c or len(c) < 2:
            continue
        k = B.norm_key(name)
        if k is None or k in pos_keys or k in have_keys:
            continue
        rows.append({"formula": name, "tc": 0.0, "n_reports": 1, "tc_iqr": 0.0,
                     "chemsys": "-".join(sorted(c)), "nel": len(c), "k": k, "weight": 1.0,
                     "mag_order": order_of(r) or "unknown"})
    d = pd.DataFrame(rows).drop_duplicates("k")
    d.to_csv(CAND, index=False)
    need = set()
    for cs in d.chemsys:
        els = sorted(cs.split("-"))
        for sub in B._subsets(els):
            need.add("-".join(sub))
    with open(SYSTEMS, "w") as f:
        f.write("\n".join(sorted(need)) + "\n")
    print(f"Cu-bearing magnetic rows: {len(rows)} raw -> {len(d)} unique compositions not already in V8 "
          f"(blocked as Tc>0 duplicates: dropped silently)", flush=True)
    print(f"  ordering: {d.mag_order.value_counts().to_dict()} | elements per formula: {d.nel.value_counts().sort_index().to_dict()}")
    print(f"  {len(need)} chemical systems -> {SYSTEMS}\n  -> {CAND}")


def cmd_pool(**_):
    B.CAND, B.SYSTEMS, B.POOL = CAND, SYSTEMS, _os.path.join(SCDB, "cuneg_pool.pickle")
    B.cmd_pool()


def cmd_match_dope(limit=None, **_):
    B.CAND, B.SYSTEMS, B.POOL = CAND, SYSTEMS, _os.path.join(SCDB, "cuneg_pool.pickle")
    B.SOURCE, B.CIFS = SOURCE, CIFS
    B.cmd_match_dope(limit=limit)


def cmd_graphs(**_):
    B.SOURCE, B.CIFS, B.IDPROP, B.GRAPHS, B.INDEX_OUT = SOURCE, CIFS, IDPROP, GRAPHS, INDEX_OUT
    B.cmd_graphs()


def cmd_merge(**_):
    """V9 = V8 + the Cu negatives (tc=0, label=1) + their descriptors/metadata."""
    v8 = pd.read_pickle(_os.path.join(MP, "SC_MP_V8_supercon.pickle"))
    neg = pd.read_pickle(INDEX_OUT + ".pickle")
    neg["graph_path"] = neg.graph_path.map(lambda p: _os.path.relpath(p, _ROOT) if _os.path.isabs(p) else p)
    neg["value"] = 0.0
    neg["tc"] = 0.0
    src = pd.read_csv(SOURCE).set_index("id")
    cand = pd.read_csv(CAND).set_index("k")
    neg["mag_order"] = neg.id.map(lambda i: cand.mag_order.get(src.k.get(i.replace(".cif", ""), None), "unknown"))
    v9 = pd.concat([v8, neg[["id", "value", "graph_path", "label", "tc"]]], ignore_index=True).drop_duplicates("id")
    v9.to_pickle(_os.path.join(MP, "SC_MP_V9_cuneg.pickle"))
    d8 = pd.read_pickle(_os.path.join(MP, "descriptors_v8_supercon.pickle"))
    sys_path = _os.path.join(_ROOT, "models", "head")
    if sys_path not in _sys.path:
        _sys.path.insert(0, sys_path)
    from descriptors import build_descriptor_table
    build_descriptor_table(INDEX_OUT + ".pickle", _os.path.join(MP, "descriptors_v9_cuneg_only.pickle"))
    dn = pd.read_pickle(_os.path.join(MP, "descriptors_v9_cuneg_only.pickle"))
    table = dict(d8["table"]); table.update(dn["table"])
    pd.to_pickle({"names": list(d8["names"]), "table": table, "failed": []},
                 _os.path.join(MP, "descriptors_v9_cuneg.pickle"))
    meta = pd.read_csv(_os.path.join(MP, "3DSC_MP_v8.csv"), low_memory=False)
    have = set(meta.cif.astype(str).map(_os.path.basename))
    new = [{"cif": i, "sc_class": "Cuprate", "synth_doped": True} for i in neg.id if i not in have]
    pd.concat([meta, pd.DataFrame(new)], ignore_index=True).to_csv(
        _os.path.join(MP, "3DSC_MP_v9.csv"), index=False)
    miss = [i for i in v9.id if i not in table]
    print(f"V9: {len(v9)} rows = V8 {len(v8)} + negatives {len(neg)} | Tc>0 {int((v9.tc>0).sum())} "
          f"({100*(v9.tc>0).mean():.1f}%) | ratio {(v9.tc>0).sum()/(v9.tc<=0).sum():.2f}:1 | descriptors missing {len(miss)}")
    import subprocess
    from pymatgen.core import Composition as C2
    v9["f"] = v9.id.str.split("-MP-").str[0].str.split("-ICSD-").str[0]
    def cu(f):
        try: return {"Cu", "O"} <= set(C2(f).get_el_amt_dict())
        except Exception: return False
    c = v9[v9.f.map(cu)]
    print(f"  Cu+O family: {len(c)} rows | Tc>0 {int((c.tc>0).sum())} | Tc=0 {int((c.tc<=0).sum())} "
          f"| local ratio {(c.tc>0).sum()/max((c.tc<=0).sum(),1):.2f}:1  (was 2.70:1)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "pool", "match-dope", "graphs", "merge"])
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    {"candidates": cmd_candidates, "pool": cmd_pool, "match-dope": cmd_match_dope,
     "graphs": cmd_graphs, "merge": cmd_merge}[a.cmd](limit=a.limit)
