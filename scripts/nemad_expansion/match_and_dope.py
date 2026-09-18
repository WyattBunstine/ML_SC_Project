"""Improved Phase 2b+3a: match each NEMAD candidate to its top-K parent structures
(tier, e_above_hull, totreldiff), then synth-dope the FIRST parent that succeeds.
Recovers candidates whose best-formula parent isn't structurally dopable."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import sys, os, warnings, heapq, pandas as pd, numpy as np
warnings.filterwarnings("ignore")
from collections import defaultdict
from pymatgen.core import Composition
from synth_dope import synth_dope_one
from pymatgen.io.cif import CifWriter
from mp_api.client import MPRester

# the shared 3DSC totreldiff matcher (formula_match.py) — was a drifting local copy
from formula_match import chem_dict, formula_similarity

K=8
pool=pd.read_pickle("database/datafiles/NE_SCDB/mp_crystal_pool.pickle"); pool["cd"]=pool["reduced"].map(chem_dict)
by_sys=defaultdict(list)
for r in pool.itertuples(): by_sys[frozenset(r.cd)].append((r.material_id,r.cd,r.eah,r.formula_pretty))
cand=pd.read_csv("database/datafiles/NE_SCDB/nemad_candidates.csv")
topk={}   # k -> list of (tier,eah,trd,mid,parent_formula)
for c in cand.itertuples():
    cd_sc=chem_dict(c.formula); S=frozenset(cd_sc); n=len(S)
    subsets=[S] if n==1 else [S]+[S-{e} for e in S]; acc=[]
    for sub in subsets:
        for mid,cd2,eah,pf in by_sys.get(sub,()):
            tier,trd=formula_similarity(cd_sc,cd2)
            if np.isnan(tier): continue
            acc.append((int(tier),round(eah,4),round(trd,5),mid,pf))
    if acc: topk[c.k]=heapq.nsmallest(K,acc)
cand=cand[cand.k.isin(topk)].copy()
print(f"candidates with >=1 accepted parent: {len(cand)}",flush=True)

# fetch structures for all unique mids across top-K
mids=sorted({t[3] for lst in topk.values() for t in lst})
key = os.environ.get("MP_API_KEY") or sys.exit("error: set MP_API_KEY (env-only)")
print(f"fetching {len(mids)} structures",flush=True)
cache={}
with MPRester(key) as mpr:
    for i in range(0,len(mids),400):
        for d in mpr.materials.summary.search(material_ids=mids[i:i+400],fields=["material_id","structure"]):
            cache[str(d.material_id)]=d.structure
        print(f"  {min(i+400,len(mids))}/{len(mids)}",flush=True)

outdir="database/datafiles/MP/cifs_v5_nemad"; os.makedirs(outdir,exist_ok=True)
rows=[]; nfail=0
info={r.k:r for r in cand.itertuples()}
for k,lst in topk.items():
    c=info[k]; done=False
    for tier,eah,trd,mid,pf in lst:
        st0=cache.get(mid)
        if st0 is None: continue
        try: st,reason=synth_dope_one(st0,c.formula)
        except Exception: continue
        if st is None: continue
        ident=f"{c.formula}-MP-{mid}"; cifp=f"{outdir}/{ident}.cif"
        try: CifWriter(st,symprec=None).write_file(cifp)
        except Exception: continue
        rows.append(dict(id=ident,cif=cifp,tc=c.tc,weight=c.weight,nemad_n=c.nemad_n,nemad_iqr=c.nemad_iqr,
                         parent_formula=pf,tier=tier,totreldiff=trd,doping=reason,eah=eah)); done=True; break
    if not done: nfail+=1
out=pd.DataFrame(rows)
out.to_csv("database/datafiles/MP/SC_MP_V5_nemad_source.csv",index=False)
out[["id","parent_formula"]].to_csv("database/datafiles/MP/SC_MP_V5_nemad_parents.csv",index=False)
print(f"\nDONE: {len(out)} synth-doped (of {len(cand)} matched cands, {100*len(out)/len(cand):.0f}%); still failed {nfail}",flush=True)
print(f"  identical {int((out.doping=='identical').sum())}  doped {int((out.doping=='doped').sum())}  SC(tc>1) {int((out.tc>1).sum())}  weighted {out.weight.sum():.0f}",flush=True)
