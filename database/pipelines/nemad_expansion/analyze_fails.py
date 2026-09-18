"""Diagnose the synth-doping failures: reason breakdown, which are oxygen-interstitial,
and how concentrated they are by parent (-> which ICSD structures would unlock the most)."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import os, sys, warnings, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
from synth_dope import synth_dope_one
from pymatgen.core import Composition
from mp_api.client import MPRester

m = pd.read_csv("database/datafiles/NE_SCDB/nemad_matches.csv")   # top-1 match per candidate (3574)
src = pd.read_csv("database/datafiles/MP/SC_MP_V5_nemad_source.csv")
succ = set(src.id.map(lambda i: i.split("-MP-")[0]))              # succeeded formulas
fail = m[~m.formula.isin(succ)].copy()
print(f"matched candidates: {len(m)} | succeeded: {len(m)-len(fail)} | FAILED: {len(fail)}", flush=True)

# fetch structures for failed candidates' top-1 parents
mids = sorted(fail.material_id.unique())
key = os.environ.get("MP_API_KEY") or sys.exit("error: set MP_API_KEY (env-only)")
cache = {}
with MPRester(key) as mpr:
    for i in range(0, len(mids), 400):
        for d in mpr.materials.summary.search(material_ids=mids[i:i+400], fields=["material_id","structure"]):
            cache[str(d.material_id)] = d.structure

def diff_elements(target, parent):
    ct = Composition(target).fractional_composition.get_el_amt_dict()
    cp = Composition(parent).fractional_composition.get_el_amt_dict()
    els = set(ct) | set(cp)
    return [e for e in els if abs(ct.get(e,0)-cp.get(e,0)) > 0.005]

reasons, rows = {}, []
for r in fail.itertuples():
    st0 = cache.get(r.material_id)
    if st0 is None: reason="no structure"
    else:
        try: st, reason = synth_dope_one(st0, r.formula); reason = "SUCCESS(now)" if st is not None else reason
        except Exception as e: reason = f"exc:{type(e).__name__}"
    reasons[reason] = reasons.get(reason,0)+1
    de = diff_elements(r.formula, r.parent_formula)
    o_only = (de == ["O"])                       # differs from parent ONLY in oxygen
    rows.append(dict(formula=r.formula, tc=r.tc, parent=r.parent_formula, material_id=r.material_id,
                     reason=reason, diff="".join(de), o_interstitial=o_only, nemad_n=r.nemad_n))
F = pd.DataFrame(rows)
print("\n=== failure reason breakdown ===")
for k,v in sorted(reasons.items(), key=lambda x:-x[1]): print(f"  {k:42s}: {v}")

oi = F[F.o_interstitial]
print(f"\n=== OXYGEN-INTERSTITIAL failures (differ from parent only in O): {len(oi)} ===")
print(f"  superconducting (tc>1): {int((oi.tc>1).sum())}   high-Tc(>77): {int((oi.tc>77).sum())}")
def fam(f):
    s=set(Composition(f).get_el_amt_dict())
    if {"Cu","O"}<=s and "Y" in s and "Ba" in s: return "YBCO (Y-Ba-Cu-O)"
    if {"Cu","O"}<=s and "Bi" in s: return "Bi-cuprate"
    if {"Cu","O"}<=s and "Tl" in s: return "Tl-cuprate"
    if {"Cu","O"}<=s and "Hg" in s: return "Hg-cuprate"
    if {"Cu","O"}<=s and "La" in s: return "La-214 (LSCO-like)"
    if {"Cu","O"}<=s: return "other cuprate"
    if {"Fe"}<=s and ({"As"}<=s or {"Se"}<=s): return "Fe-based"
    return "other"
oi["family"]=oi.parent.map(fam)
print("\n  by family:")
for k,v in oi.family.value_counts().items(): print(f"    {k:24s}: {v}")
print("\n  === top recurring PARENTS (fetch these from ICSD to unlock the most) ===")
g=oi.groupby("parent").agg(n_candidates=("formula","size"),n_SC=("tc",lambda x:(x>1).sum()),
                           max_tc=("tc","max"),chemsys=("parent",lambda s:"-".join(sorted(Composition(s.iloc[0]).get_el_amt_dict())))).reset_index()
for r in g.sort_values("n_candidates",ascending=False).head(18).itertuples():
    print(f"    {r.parent:20s} [{r.chemsys:16s}] unlocks {int(r.n_candidates):3d} candidates ({int(r.n_SC)} SC, max_tc {r.max_tc:.0f})")
F.to_csv("database/datafiles/MP/nemad_doping_failures.csv",index=False)
print(f"\nfull failure table -> database/datafiles/MP/nemad_doping_failures.csv")
# concentration
top10 = g.sort_values("n_candidates",ascending=False).head(10).n_candidates.sum()
print(f"top-10 parents cover {top10}/{len(oi)} O-interstitial failures ({100*top10/max(len(oi),1):.0f}%)")
