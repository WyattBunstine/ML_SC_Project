"""Match NEMAD candidates against the fetched MP crystal pool (20k materials).
Faithful 3DSC formula matcher; best match by (e_above_hull, totreldiff)."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from collections import defaultdict
from pymatgen.core import Composition

# the shared 3DSC totreldiff matcher (formula_match.py) — was a drifting local copy
from formula_match import chem_dict, formula_similarity

pool=pd.read_pickle("database/datafiles/NE_SCDB/mp_crystal_pool.pickle")
pool["cd"]=pool["reduced"].map(chem_dict)
by_sys=defaultdict(list)
for r in pool.itertuples():
    by_sys[frozenset(r.cd)].append((r.material_id,r.cd,r.eah,r.formula_pretty))
print(f"pool: {len(pool)} materials, {len(by_sys)} systems",file=sys.stderr)

cand=pd.read_csv("database/datafiles/NE_SCDB/nemad_candidates.csv")
rows=[]
for c in cand.itertuples():
    cd_sc=chem_dict(c.formula); S=frozenset(cd_sc); n=len(S)
    subsets=[S] if n==1 else [S]+[S-{e} for e in S]
    best=None
    for sub in subsets:
        for mid,cd2,eah,pf in by_sys.get(sub,()):
            tier,trd=formula_similarity(cd_sc,cd2)
            if np.isnan(tier): continue
            # tier-first (formula closeness) then stability then trd: prefer the exact
            # SC phase (e.g. YBCO-123) over a more-stable but wrong phase (YBCO-124)
            key=(int(tier),round(eah,4),round(trd,5))
            if best is None or key<best[0]:
                best=(key,tier,trd,mid,pf,eah)
    if best is not None:
        rows.append(dict(formula=c.formula,tc=c.tc,nemad_n=c.nemad_n,nemad_iqr=c.nemad_iqr,
                         weight=c.weight,chemsys=c.chemsys,k=c.k,material_id=best[3],
                         parent_formula=best[4],totreldiff=round(best[2],4),tier=int(best[1]),
                         e_above_hull=round(best[5],4)))
out=pd.DataFrame(rows)
out.to_csv("database/datafiles/NE_SCDB/nemad_matches.csv",index=False)
print(f"MATCHED {len(out)}/{len(cand)} ({100*len(out)/len(cand):.0f}%)")
print("  tier: identical %d  similar %d  doped %d"%((out.tier==1).sum(),(out.tier==2).sum(),(out.tier==3).sum()))
print("  SC matched (tc>1): %d   |  unique parent structures: %d"%((out.tc>1).sum(),out.material_id.nunique()))
print("  weighted new rows (sum weight): %.0f   median trd %.4f"%(out.weight.sum(),out.totreldiff.median()))
unm=cand[~cand.k.isin(out.k)]
print("  unmatched: %d (of which SC tc>1: %d)"%(len(unm),(unm.tc>1).sum()))
print("\n=== spot-check canonical parents (should now be correct phase) ===")
for f in ["YBa2Cu3O7","YBa2Cu3O6.5","La1.85Sr0.15CuO4","MgB2","Ba0.6K0.4BiO3","FeSe","Nb3Ge"]:
    r=out[out.formula==f]
    if len(r): x=r.iloc[0]; print(f"  {f:20s} -> {x.material_id:12s} {x.parent_formula:16s} trd={x.totreldiff:.3f} tier{int(x.tier)} eah={x.e_above_hull:.3f}")
    else: print(f"  {f:20s} -> (no exact candidate row / unmatched)")
PY=1
