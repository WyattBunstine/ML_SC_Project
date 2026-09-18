"""Curated Fe-OXIDE Tc=0 negatives (user, 2026-09-14): counter-evidence for the
iron-oxide false positives (Fe-based Tc=0 rows predicted 8-9 K under V8/V9).

Candidates: Fe-bearing, O-bearing compositions in the NEMAD magnetic set
(measured Curie/Neel order = measured non-SC) with NO pnictogen/chalcogen (those
are the FeAs/FeSe superconductor manifold and would suppress it) and NO Cu
(handled by the Cu tier), not already in V10, and with no Tc>0 SuperCon report.
Same matcher/doper as the SuperCon expansion; parent gate requires Fe.
V11 = V10 + these rows (tc=0, label=1, family Oxide).

  candidates | pool | match-dope | graphs | merge
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__)); _ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
for _p in (_ROOT, _HERE, _os.path.join(_ROOT, "database", "pipelines", "nemad_expansion")):
    if _p not in _sys.path: _sys.path.insert(0, _p)
import argparse, re, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition
import build_supercon_v7 as B
MP, SCDB = B.MP, B.SCDB
RAW = _os.path.join(_ROOT, "database", "datafiles", "NE_SCDB", "magnetic_materials.csv")
CAND = _os.path.join(SCDB, "feox_candidates.csv"); SYSTEMS = _os.path.join(SCDB, "feox_systems.txt"); POOL = _os.path.join(SCDB, "feox_pool.pickle")
SOURCE = _os.path.join(MP, "SC_MP_V11_feox_source.csv"); CIFS = _os.path.join(MP, "cifs_v11_feox")
IDPROP = _os.path.join(SCDB, "id_prop_v11.csv"); GRAPHS = _os.path.join(MP, "graphs_v4_v11_feox"); INDEX_OUT = _os.path.join(MP, "SC_MP_V11_feox")
ANION = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "H", "D"}


def order_of(row):
    for c, lab in (("Neel", "AFM"), ("Neel(Tn)", "AFM"), ("Curie", "FM"), ("Curie(Tc)", "FM")):
        v = row.get(c)
        if isinstance(v, str) and re.search(r"\d", v): return lab
        if isinstance(v, (int, float)) and not pd.isna(v): return lab
    return None


def cmd_candidates(**_):
    raw = pd.read_csv(RAW, low_memory=False)
    v10 = pd.read_pickle(_os.path.join(MP, "SC_MP_V10.pickle"))
    have = {B.norm_key(str(i).split("-MP-")[0].split("-ICSD-")[0]) for i in v10.id} - {None}
    sc = pd.read_csv(_os.path.join(MP, "SuperCon_Stanev2018.csv"))
    pos = {B.norm_key(str(n)) for n in sc[sc.Tc > 0].name} - {None}
    banned = {l.strip() for l in open(B.BANNED)} if _os.path.exists(B.BANNED) else set()
    rows, seen = [], set()
    for r in raw.to_dict("records"):
        name = str(r.get("Material_Name", "")).strip()
        if "Fe" not in name: continue
        try: c = Composition(name).get_el_amt_dict()
        except Exception: continue
        e = set(c)
        if "Fe" not in e or "O" not in e or "Cu" in e or e & {"As", "P", "Se", "Te", "S"}: continue
        if len(e) < 2 or max(c.values()) > 60: continue
        k = B.norm_key(name)
        if k is None or k in have or k in pos or k in banned or k in seen: continue
        seen.add(k)
        rows.append({"formula": name, "tc": 0.0, "n_reports": 1, "tc_iqr": 0.0, "chemsys": "-".join(sorted(e)),
                     "nel": len(e), "k": k, "weight": 1.0, "mag_order": order_of(r) or "unknown"})
    d = pd.DataFrame(rows); d.to_csv(CAND, index=False)
    need = set()
    for cs in d.chemsys:
        els = sorted(cs.split("-"))
        for sub in B._subsets(els): need.add("-".join(sub))
        for x in els: need.add(x)
    open(SYSTEMS, "w").write("\n".join(sorted(need)) + "\n")
    print(f"Fe-oxide magnetic candidates: {len(d)} unique compositions | ordering {d.mag_order.value_counts().to_dict()} | "
          f"elements/formula {d.nel.value_counts().sort_index().to_dict()} | {len(need)} systems", flush=True)


def _bind():
    B.CAND, B.SYSTEMS, B.POOL, B.SOURCE, B.CIFS, B.IDPROP, B.GRAPHS, B.INDEX_OUT = CAND, SYSTEMS, POOL, SOURCE, CIFS, IDPROP, GRAPHS, INDEX_OUT


def cmd_pool(**_): _bind(); B.cmd_pool()
def cmd_match_dope(limit=None, **_): _bind(); B.cmd_match_dope(limit=limit)
def cmd_graphs(**_): _bind(); B.cmd_graphs()


def cmd_merge(**_):
    """V11 = V10 + Fe-oxide negatives, minus any negative that is majority-foreign to its parent."""
    v10 = pd.read_pickle(_os.path.join(MP, "SC_MP_V10.pickle"))
    neg = pd.read_pickle(INDEX_OUT + ".pickle")
    neg["graph_path"] = neg.graph_path.map(lambda p: _os.path.relpath(p, _ROOT) if _os.path.isabs(p) else p)
    neg["value"] = 0.0; neg["tc"] = 0.0
    src = pd.read_csv(SOURCE); src["id"] = src.id.map(lambda i: i if i.endswith(".cif") else i + ".cif"); src = src.set_index("id")
    keep = []
    for i in neg.id:
        s = src.loc[i] if i in src.index else None
        if s is None: continue
        try:
            c = Composition(str(i).split("-MP-")[0]).get_el_amt_dict(); pc = set(Composition(str(s.parent_formula)).get_el_amt_dict())
        except Exception: keep.append(i); continue
        cats = {e: v for e, v in c.items() if e not in ANION}; tot = sum(cats.values())
        foreign = sum(v for e, v in cats.items() if e not in pc) / tot if tot else 0
        if not (str(s.doping).startswith("multi_sublattice") and foreign > 0.5): keep.append(i)
    dropped = len(neg) - len(keep); neg = neg[neg.id.isin(keep)]
    v11 = pd.concat([v10, neg[["id", "value", "graph_path", "label", "tc"]]], ignore_index=True).drop_duplicates("id")
    v11.to_pickle(_os.path.join(MP, "SC_MP_V11.pickle")); v11.to_csv(_os.path.join(MP, "SC_MP_V11.csv"), index=False)
    sys_path = _os.path.join(_ROOT, "models", "head")
    if sys_path not in _sys.path: _sys.path.insert(0, sys_path)
    from descriptors import build_descriptor_table
    build_descriptor_table(INDEX_OUT + ".pickle", _os.path.join(MP, "descriptors_v11_feox_only.pickle"))
    d10 = pd.read_pickle(_os.path.join(MP, "descriptors_v10.pickle")); dn = pd.read_pickle(_os.path.join(MP, "descriptors_v11_feox_only.pickle"))
    table = dict(d10["table"]); table.update({k: v for k, v in dn["table"].items() if k in set(neg.id)})
    pd.to_pickle({"names": list(d10["names"]), "table": table, "failed": []}, _os.path.join(MP, "descriptors_v11.pickle"))
    meta = pd.read_csv(_os.path.join(MP, "3DSC_MP_v10.csv"), low_memory=False)
    have = set(meta.cif.astype(str).map(_os.path.basename))
    new = [{"cif": i, "sc_class": "Oxide", "synth_doped": True} for i in neg.id if i not in have]
    pd.concat([meta, pd.DataFrame(new)], ignore_index=True).to_csv(_os.path.join(MP, "3DSC_MP_v11.csv"), index=False)
    # holdout lists: carry V10's, add any negative sharing a parent_comp group with a holdout
    from models.head.HeadData import parent_composition_groups
    ids = list(v11.id); grp = dict(zip(ids, parent_composition_groups(ids)))
    for src_, dst in (("la_series_both_holdout_ids_v10.csv", "la_series_both_holdout_ids_v11.csv"), ("nickelate_holdout_ids_v10.csv", "nickelate_holdout_ids_v11.csv")):
        h = pd.read_csv(_os.path.join(MP, src_)).id.tolist(); gs = {grp[i] for i in h if i in grp}
        add = [i for i in neg.id if grp.get(i) in gs]
        pd.DataFrame({"id": h + add}).to_csv(_os.path.join(MP, dst), index=False); print(f"  {dst}: {len(h)} -> {len(h) + len(add)}")
    miss = [i for i in v11.id if i not in table]
    print(f"V11: {len(v11)} rows = V10 {len(v10)} + Fe-oxide negatives {len(neg)} (dropped {dropped} majority-foreign) | Tc>0 {int((v11.tc > 0).sum())} "
          f"({100 * (v11.tc > 0).mean():.1f}%) | ratio {(v11.tc > 0).sum() / (v11.tc <= 0).sum():.2f}:1 | descriptors missing {len(miss)}")
    v11["f"] = v11.id.str.split("-MP-").str[0]
    def feo(f):
        try: e = set(Composition(f).get_el_amt_dict()); return "Fe" in e and "O" in e and not e & {"As", "P", "Se", "Te", "S"}
        except Exception: return False
    fo = v11[v11.f.map(feo)]
    print(f"  Fe-oxide rows: {len(fo)} | Tc>0 {int((fo.tc > 0).sum())} | Tc=0 {int((fo.tc <= 0).sum())} (was 407 / 181 / 226)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["candidates", "pool", "match-dope", "graphs", "merge"]); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    {"candidates": cmd_candidates, "pool": cmd_pool, "match-dope": cmd_match_dope, "graphs": cmd_graphs, "merge": cmd_merge}[a.cmd](limit=a.limit)
