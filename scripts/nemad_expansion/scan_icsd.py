
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)                                     # sibling modules (e.g. synth_dope)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_HERE))) # repo root (database.*, models.*)
# 3DSC repo (for synthetic_doping internals); override with THREEDSC_REPO env var.
THREEDSC_REPO = _os.environ.get("THREEDSC_REPO", _os.path.expanduser("~/Downloads/old_files/3DSC-main"))
import warnings, glob, os, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Composition, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def read_cif(p):
    try: return Structure.from_file(p)
    except Exception:
        txt="\n".join(l for l in open(p).read().splitlines())
        # strip a stray leading non-cif line (e.g. "Ok")
        if not txt.lstrip().startswith(("#","data_","_")):
            txt="\n".join(txt.splitlines()[1:])
        return Structure.from_str(txt, fmt="cif")

def dope_oxygen(structure, target_formula):
    st=structure.copy(); st.remove_oxidation_states()
    tg=Composition(target_formula).get_el_amt_dict(); cur=st.composition.get_el_amt_dict()
    cats=[e for e in tg if e!="O" and e in cur]
    if not cats: return None,"no shared cation"
    ref=max(cats,key=lambda e:cur[e]); scale=cur[ref]/tg[ref]
    des={e:v*scale for e,v in tg.items()}
    for e in set(des)|set(cur):
        if e=="O": continue
        if abs(des.get(e,0)-cur.get(e,0))>0.05: return None,"cation-codoped"
    o=[i for i,s in enumerate(st) if any(el.symbol=="O" for el in s.species)]
    occ=lambda i:sum(x for el,x in st[i].species.items() if el.symbol=="O")
    var=[i for i in o if occ(i)<0.999]; fixed=sum(occ(i) for i in o if occ(i)>=0.999)
    chain=des.get("O",0)-fixed
    if chain<-1e-6 or chain>len(var)+1e-6: return None,"O unphysical"
    return st,"O-doped"

def family(comp):
    s=set(comp.get_el_amt_dict()); d=comp.get_el_amt_dict()
    re_=s&{"Y","Nd","Sm","Er","Gd","Ho","Tm","Eu","La","Dy","Yb","Pr"}
    if {"Cu","Ba"}<=s and re_ and abs(d.get("Cu",0)/max(min(d[r] for r in re_),.1)-3)<1.2: return f"{sorted(re_)[0]}-123"
    if {"Cu","Bi"}<=s: return "Bi-cuprate"
    if {"Cu","Hg"}<=s: return "Hg-cuprate"
    if {"Cu","Tl"}<=s: return "Tl-cuprate"
    if {"Cu","La"}<=s and "Ba" not in s: return "La-214"
    return "other"

det=pd.read_csv("database/datafiles/MP/ICSD_pull_targets_detail.csv")
rows=[]
for p in sorted(glob.glob("database/datafiles/MP/ICSD_Parent_Cifs/*.cif")):
    code=os.path.basename(p).replace("EntryWithCollCode","").replace(".cif","")
    try: st=read_cif(p)
    except Exception as e: print(f"{code}: READ FAIL {e}"); continue
    comp=st.composition; red=comp.reduced_formula
    try: spg=SpacegroupAnalyzer(st,symprec=0.01).get_space_group_symbol()
    except Exception: spg="?"
    o=[i for i,s in enumerate(st) if any(el.symbol=="O" for el in s.species)]
    occ=lambda i:sum(x for el,x in st[i].species.items() if el.symbol=="O")
    part=[round(occ(i),2) for i in o if occ(i)<0.999]
    O=comp.get_el_amt_dict().get("O",0)
    fam=family(comp)
    rows.append(dict(code=code,formula=red,fam=fam,spg=spg,natoms=len(st),
                     Ototal=round(O,2),n_partial_O=len(part),partial_occs=str(part),
                     ordered=st.is_ordered))
S=pd.DataFrame(rows)
pd.set_option("display.width",240,"display.max_colwidth",34)
print(S[["code","formula","fam","spg","Ototal","n_partial_O","partial_occs"]].to_string(index=False))

print("\n=== DUPLICATES (same family) -> best pick ===")
for fam,g in S.groupby("fam"):
    if len(g)>1:
        # best = oxygen-rich WITH partial O sites (dopable), then most O, then most partial sites
        g=g.assign(dopable=g.n_partial_O>0).sort_values(["dopable","Ototal","n_partial_O"],ascending=[False,False,False])
        best=g.iloc[0]
        print(f"  {fam}: {list(g.code)}  -> BEST {best.code} ({best.formula}, O{best.Ototal}, {best.n_partial_O} partial-O sites)")
        for _,r in g.iterrows():
            tag="  <-- best" if r.code==best.code else ""
            print(f"       {r.code}: O{r.Ototal}, {r.n_partial_O} partial-O sites, spg {r.spg}{tag}")

print("\n=== doping verification (against real targets per family) ===")
for r in S.itertuples():
    st=read_cif(f"database/datafiles/MP/ICSD_Parent_Cifs/EntryWithCollCode{r.code}.cif")
    tg=det[det.proto.str.startswith(r.fam.split('-')[0]) if r.fam!='other' else det.proto=='zzz']
    # match targets whose framework shares the family cations
    fam_cats=set(Composition(r.formula).get_el_amt_dict())-{"O"}
    cand=det[det.formula.map(lambda f: (set(Composition(str(f)).get_el_amt_dict())-{"O"})==fam_cats or (set(Composition(str(f)).get_el_amt_dict())-{"O"}).issubset(fam_cats|{"Ca"}))]
    ok=cod=bad=0
    for t in cand.itertuples():
        _,why=dope_oxygen(st,t.formula)
        if why=="O-doped": ok+=1
        elif why=="cation-codoped": cod+=1
        else: bad+=1
    print(f"  {r.code} [{r.fam:11s} {r.formula:16s}]: {ok:3d} pure-O dope OK | {cod:3d} co-doped | {bad:2d} unphysical  (of {len(cand)} family targets)")
S.to_csv("database/datafiles/MP/ICSD_parent_scan.csv",index=False)
