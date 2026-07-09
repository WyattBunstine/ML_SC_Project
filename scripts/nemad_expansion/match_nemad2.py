"""Match NEMAD candidates against the fetched MP crystal pool (20k materials).
Faithful 3DSC formula matcher; best match by (e_above_hull, totreldiff)."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import pandas as pd, numpy as np, warnings, sys
warnings.filterwarnings("ignore")
from collections import defaultdict
from pymatgen.core import Composition

LMR,LTR,LMA=0.10001,0.05001,0.15001
HMR,HTR,HMA=0.20001,0.15001,0.3001

def chem_dict(comp):
    d=Composition(comp).get_el_amt_dict() if isinstance(comp,str) else dict(comp)
    els=sorted([e for e in d if e!="O"])+(["O"] if "O" in d else [])
    return {e:float(d[e]) for e in els}

def formula_similarity(cd_sc,cd_2):
    els_sc,els_2=list(cd_sc),list(cd_2)
    if len(els_sc)<len(els_2) or not all(e in els_sc for e in els_2): return np.nan,np.nan
    if len(els_sc)>len(els_2) and len(els_2)==1: return np.nan,np.nan
    q_sc=np.array([cd_sc[e] for e in els_sc]); q_2=np.array([cd_2.get(e,0.0) for e in els_sc])
    norm=q_sc.sum()/q_2.sum(); q_2=q_2*norm
    diffs=np.abs(q_sc-q_2); reldiffs=2*diffs/(q_sc+q_2)
    trd=2*diffs.sum()/(q_sc.sum()+q_2.sum())
    def ok(mr,tr,ma):
        if trd>tr: return False
        return not any((d>ma and rd>mr) for d,rd in zip(diffs,reldiffs))
    if trd==0: return 1,trd
    if ok(LMR,LTR,LMA): return 2,trd
    if ok(HMR,HTR,HMA): return 3,trd
    return np.nan,trd

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
