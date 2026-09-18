
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))
import warnings, glob, os, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def read_cif(p):
    try: return Structure.from_file(p)
    except Exception:
        txt=open(p).read()
        if not txt.lstrip().startswith(("#","data_","_")): txt="\n".join(txt.splitlines()[1:])
        return Structure.from_str(txt, fmt="cif")

RE={"Y","Nd","Sm","Er","Gd","Ho","Tm","Eu","La","Dy","Yb","Pr","Ce","Tb","Lu"}
def fam(comp):
    s=set(comp.get_el_amt_dict()); d=comp.get_el_amt_dict()
    re_=s&RE
    if {"Cu","Ba"}<=s and re_ and (d.get("Cu",0)>=2.5):
        # RE-123 (incl. La-123 solid soln). pick the RE
        return "RE123:"+sorted(re_,key=lambda r:-d[r])[0]
    if {"Cu","Bi"}<=s: return "Bi-cuprate"
    if {"Cu","Hg"}<=s: return "Hg-cuprate"
    if {"Cu","Tl"}<=s: return "Tl-cuprate"
    if {"Cu","La"}<=s and "Ba" not in s: return "La-214"
    return "other:"+comp.reduced_formula

# --- inventory ---
inv=[]
for p in sorted(glob.glob("database/datafiles/MP/ICSD_Parent_Cifs/*.cif")):
    code=os.path.basename(p).replace("EntryWithCollCode","").replace(".cif","")
    try: st=read_cif(p)
    except Exception as e: print("READ FAIL",code,e); continue
    c=st.composition; o=[i for i,s in enumerate(st) if any(e.symbol=="O" for e in s.species)]
    occ=lambda i:sum(x for e,x in st[i].species.items() if e.symbol=="O")
    npart=sum(1 for i in o if occ(i)<0.999)
    try: spg=SpacegroupAnalyzer(st,symprec=0.05).get_space_group_symbol()
    except Exception: spg="?"
    ortho = spg.startswith("P m m m") or ("m m m" in spg and not spg.startswith("P 4"))
    inv.append(dict(code=code,formula=c.reduced_formula,fam=fam(c),spg=spg,O=round(c.get_el_amt_dict().get("O",0),2),
                    npart=npart,natoms=len(st),ortho=ortho))
I=pd.DataFrame(inv)
print("=== INVENTORY ===")
print(I.sort_values("fam")[["code","formula","fam","spg","O","npart","natoms"]].to_string(index=False))

# a good RE-123 substitution template = orthorhombic-ish 123 with >=1 partial chain-O
templ = I[(I.fam.str.startswith("RE123"))&(I.npart>0)].sort_values(["ortho","npart"],ascending=False)
have_template = len(templ)>0
print(f"\nRE-123 substitution template available: {have_template}",
      f"-> best: {templ.iloc[0].code} ({templ.iloc[0].formula}, {templ.iloc[0].spg}, {int(templ.iloc[0].npart)} partial-O)" if have_template else "")

# --- required families from the O-interstitial targets ---
det=pd.read_csv("database/datafiles/MP/ICSD_pull_targets_detail.csv")
def tgt_fam(f): return fam(Composition(str(f)))
det["need"]=det.formula.map(tgt_fam)
req=det.groupby("need").agg(targets=("formula","size"),SC=("tc",lambda x:(x>1).sum()),max_tc=("tc","max")).reset_index().sort_values("targets",ascending=False)

native_fams=set(I.fam)
def covered(need):
    if need in native_fams: return "native"
    if need.startswith("RE123") and have_template: return "via RE-substitution"
    return "MISSING"
req["coverage"]=req.need.map(covered)
print("\n=== COVERAGE of flagged target families ===")
tot=cov=0
for r in req.itertuples():
    tot+=r.targets
    if r.coverage!="MISSING": cov+=r.targets
    print(f"  {r.need:18s} targets={int(r.targets):4d}  maxTc={r.max_tc:3.0f}   {r.coverage}")
print(f"\ncovered targets: {cov}/{tot} ({100*cov/tot:.0f}%)")
miss=req[req.coverage=="MISSING"]
if len(miss):
    print("\n=== STILL MISSING (fetch these) ===")
    for r in miss.itertuples():
        print(f"  {r.need:18s}: {int(r.targets)} targets (up to {r.max_tc:.0f} K)")
else:
    print("\n=== ALL flagged families covered ===")
