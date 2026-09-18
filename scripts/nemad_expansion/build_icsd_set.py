"""Phase 6+7a: assign each O-interstitial NEMAD target to its ICSD parent (RE-substituting
the 123 template as needed), dope it with the combined doper, write CIFs + build-db source."""

import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))

import warnings, os, sys, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition
from pymatgen.io.cif import CifWriter
from database.icsd_doping import read_icsd_cif, re_substitute, combined_dope

CIF="database/datafiles/MP/ICSD_Parent_Cifs/EntryWithCollCode%s.cif"
RE_SET={"Y","Nd","Sm","Er","Gd","Ho","Tm","Eu","Dy","Pr","Yb","Tb","Lu","La"}
# family -> best parent cif (validated picks)
PARENT={"Pb":"86957","Bi":"173899","Hg":"80718","Tl":"202628","La214":"73486","Fe1111":"167338"}
RE123_TEMPLATE="63321"   # Y-123 Pmmm, partial chain-O; RE-substituted per target

def assign(formula):
    """Return (parent_cif_code, re_to_substitute_or_None, family)."""
    d=Composition(str(formula)).get_el_amt_dict(); s=set(d)
    if {"Pb","Cu"}<=s: return PARENT["Pb"],None,"Pb-3212"
    if {"Bi","Cu"}<=s: return PARENT["Bi"],None,"Bi-cuprate"
    if {"Hg","Cu"}<=s: return PARENT["Hg"],None,"Hg-cuprate"
    if {"Tl","Cu"}<=s: return PARENT["Tl"],None,"Tl-cuprate"
    re=[e for e in s if e in RE_SET]
    if {"Ba","Cu"}<=s and re and d.get("Cu",0)>=2.5:
        RE=max(re,key=lambda e:d[e])
        return RE123_TEMPLATE,(None if RE=="Y" else RE),f"{RE}-123"
    if {"La","Cu"}<=s and "Ba" not in s and d.get("Cu",0)>0: return PARENT["La214"],None,"La-214"
    if {"Fe","As"}<=s: return PARENT["Fe1111"],None,"Fe-1111"
    return None,None,"unassigned"

# targets: O-interstitial set + weights
det=pd.read_csv("database/datafiles/MP/ICSD_pull_targets_detail.csv")
cand=pd.read_csv("database/datafiles/NE_SCDB/nemad_candidates.csv")
wmap=dict(zip(cand.formula,cand.weight)); nmap=dict(zip(cand.formula,cand.nemad_n)); imap=dict(zip(cand.formula,cand.nemad_iqr))

# cache parents (+RE-subbed)
raw={}
def get_parent(code,re):
    key=(code,re)
    if key not in raw:
        st=read_icsd_cif(CIF%code)
        raw[key]=re_substitute(st,re) if re else st
    return raw[key]

outdir="database/datafiles/MP/cifs_v5_icsd"; os.makedirs(outdir,exist_ok=True)
rows=[]; byfam={}; failfam={}
for r in det.itertuples():
    code,re,fam=assign(r.formula)
    byfam.setdefault(fam,[0,0]); byfam[fam][1]+=1
    if code is None: failfam[fam]=failfam.get(fam,0)+1; continue
    try:
        st,why=combined_dope(get_parent(code,re),r.formula)
    except Exception as e: st,why=None,f"exc:{type(e).__name__}"
    if st is None: continue
    byfam[fam][0]+=1
    ident=f"{r.formula}-ICSD-{code}"
    cifp=f"{outdir}/{ident}.cif"
    try: CifWriter(st,symprec=None).write_file(cifp)
    except Exception: continue
    parent_formula=Composition(get_parent(code,re).composition.reduced_formula).reduced_formula
    rows.append(dict(id=ident,cif=ident+".cif",tc=r.tc,weight=wmap.get(r.formula,0.3),
                     nemad_n=nmap.get(r.formula),nemad_iqr=imap.get(r.formula),
                     parent_formula=parent_formula,family=fam,source_cif=code))
out=pd.DataFrame(rows)
out.to_csv("database/datafiles/MP/SC_MP_V5_icsd_source.csv",index=False)
print(f"ICSD-backed synth-doped: {len(out)} / {len(det)} O-interstitial targets ({100*len(out)/len(det):.0f}%)")
print(f"  superconducting (tc>1): {int((out.tc>1).sum())}   high-Tc(>77): {int((out.tc>77).sum())}")
print("\nby family (doped / total):")
for f in sorted(byfam,key=lambda x:-byfam[x][1]):
    print(f"  {f:14s}: {byfam[f][0]:3d}/{byfam[f][1]:3d}"+(f"   [{failfam[f]} unassigned]" if f in failfam else ""))
