"""Data-quality sweep over the expanded index (V9 = V45 + SuperCon expansion +
ICSD-parent rows + Cu-oxide negatives). Every row is run through nine checks;
anything flagged lands in docs/data_curation/v9_audit_suspects.csv with the
reason(s), ranked so the worst rows come first.

  A  formula anomalies      coefficient > 60, > 300 atoms, or a trace dopant < 0.005
  B  Tc vs chemistry        alloy > 25 K, elemental > 10 K, hydride > 40 K (pressure
                            phases), Fe-based > 60 K, cuprate > 140 K, anything > 165 K
  C  charge balance         formal Cu oxidation outside [1.4, 3.3] on Cu-oxides
  D  conflicting duplicates same normalized composition, |dTc| > max(5 K, 30 %)
  E  literature conflict    SuperCon multi-report rows whose IQR > max(5 K, 50 %)
  F  mislabeled negatives   a Tc=0 negative whose composition has a Tc>0 SuperCon report
  G  parent mismatch        built on a parent lacking the family's redox centre, or
                            whose anion/cation ratio differs from the target by > 0.25
  H  composition fidelity   the written structure's composition differs from the label
  I  Tc units               0 < Tc < 0.05 K (mK?) or Tc > 165 K

  python database/pipelines/supercon_expansion/audit_v9.py
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE)))
for _p in (_ROOT, _os.path.join(_ROOT, "scripts")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
warnings.filterwarnings("ignore")
from pymatgen.core import Composition, Structure  # noqa: E402
from dome_stats import OX, cu_oxidation  # noqa: E402

MP = _os.path.join(_ROOT, "database", "datafiles", "MP")
OUT = _os.path.join(_ROOT, "docs", "data_curation", "v9_audit_suspects.csv")
ANION = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "H", "D"}
TM = {"Sc", "Y", "La", "Lu", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Tc", "Re",
      "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt", "Cu", "Ag", "Au", "Zn", "Cd", "Hg",
      "Al", "Ga", "In", "Sn", "Pb", "Bi", "Sb", "Ge", "Si", "Mg", "Be", "Li", "Na", "K", "Ca",
      "Sr", "Ba", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Th", "U"}


def formula(i):
    return str(i).split("-MP-")[0].split("-ICSD-")[0].replace("-synth_doped", "")


def key(f):
    try:
        c = Composition(f).fractional_composition.get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return None
    return "|".join(f"{e}{round(v, 3)}" for e, v in sorted(c.items()))


def cls(c):
    e = set(c)
    if e <= TM and not (e & ANION):
        return "elemental" if len(e) == 1 else "alloy"
    if "Cu" in e and "O" in e:
        return "cuprate"
    if "Fe" in e and (e & {"As", "P", "Se", "Te", "S"}):
        return "fe-based"
    if ("H" in e or "D" in e) and "O" not in e:
        return "hydride"
    return "other"


def ratio(c):
    a = sum(v for e, v in c.items() if e in ANION); k = sum(v for e, v in c.items() if e not in ANION)
    return a / k if k else np.nan


def main():
    v9 = pd.read_pickle(_os.path.join(MP, "SC_MP_V9_cuneg.pickle"))
    v45 = set(pd.read_pickle(_os.path.join(MP, "SC_MP_V4_doped_v45.pickle")).id)
    v9["formula"] = v9.id.map(formula)
    v9["origin"] = np.where(v9.id.isin(v45), "V45", "new")
    src = pd.concat([pd.read_csv(_os.path.join(MP, "SC_MP_V7_supercon_source.csv")),
                     pd.read_csv(_os.path.join(MP, "SC_MP_V9_cuneg_source.csv"))], ignore_index=True)
    src["id"] = src.id.map(lambda i: i if i.endswith(".cif") else i + ".cif")
    src = src.drop_duplicates("id").set_index("id")
    neg_ids = set(pd.read_csv(_os.path.join(MP, "SC_MP_V9_cuneg_source.csv")).id.map(lambda i: i if i.endswith(".cif") else i + ".cif"))
    cand = pd.read_csv(_os.path.join(_ROOT, "database", "datafiles", "SC_EXPAND", "supercon_candidates.csv")).set_index("k")
    sc = pd.read_csv(_os.path.join(MP, "SuperCon_Stanev2018.csv"))
    sc["k"] = sc.name.map(key)
    sc_pos = sc[sc.Tc > 0].groupby("k").Tc.max()
    flags = defaultdict(list)
    comps = {}
    for r in v9.itertuples():
        try:
            c = Composition(r.formula).get_el_amt_dict()
        except Exception:  # noqa: BLE001
            flags[r.id].append("A:unparseable"); continue
        comps[r.id] = c
        tot = sum(c.values())
        if max(c.values()) > 60 or tot > 300:
            flags[r.id].append(f"A:huge-coefficient(max {max(c.values()):.0f}, {tot:.0f} atoms)")
        if 0 < min(c.values()) / tot < 0.005:
            flags[r.id].append(f"A:trace-dopant({min(c, key=c.get)} {min(c.values())/tot:.4f})")
        k = cls(c)
        lim = {"alloy": 25, "elemental": 10, "hydride": 40, "fe-based": 60, "cuprate": 140}.get(k, 165)
        if r.tc > lim:
            flags[r.id].append(f"B:{k}-Tc-{r.tc:.0f}K>{lim}")
        if r.tc > 165:
            flags[r.id].append(f"I:Tc-{r.tc:.0f}K")
        if 0 < r.tc < 0.05:
            flags[r.id].append(f"I:Tc-{r.tc:.3f}K(mK?)")
        if k == "cuprate":
            ox = cu_oxidation(r.formula)
            if ox is not None and not (1.4 <= ox <= 3.3):
                flags[r.id].append(f"C:Cu-ox-{ox:.2f}")
        if r.id in src.index:
            s = src.loc[r.id]
            try:
                pc = Composition(str(s.parent_formula)).get_el_amt_dict()
            except Exception:  # noqa: BLE001
                pc = None
            if pc is not None:
                centre = {"cuprate": "Cu", "fe-based": "Fe"}.get(k)
                if centre and centre not in pc:
                    flags[r.id].append(f"G:parent-lacks-{centre}({s.parent_formula})")
                dr = ratio(c) - ratio(pc)
                if not np.isnan(dr) and abs(dr) > 0.25 and k != "alloy":
                    flags[r.id].append(f"G:anion-ratio-{dr:+.2f}({s.parent_formula})")
            kk = s.k if isinstance(s.k, str) else None
            if kk in cand.index and cand.loc[kk, "n_reports"] >= 2:
                iq, tc = cand.loc[kk, "tc_iqr"], cand.loc[kk, "tc"]
                if iq > max(5.0, 0.5 * tc):
                    flags[r.id].append(f"E:literature-IQR-{iq:.1f}K(n={int(cand.loc[kk,'n_reports'])})")
        if r.id in neg_ids:
            kk = key(r.formula)
            if kk in sc_pos.index:
                flags[r.id].append(f"F:negative-but-SuperCon-Tc-{sc_pos[kk]:.1f}K")
    # D: conflicting duplicates
    v9["k"] = v9.formula.map(key)
    for kk, g in v9.groupby("k"):
        if len(g) < 2:
            continue
        lo, hi = g.tc.min(), g.tc.max()
        if hi - lo > max(5.0, 0.3 * hi):
            for i in g.id:
                flags[i].append(f"D:duplicate-Tc-{lo:.1f}..{hi:.1f}K(n={len(g)})")
    # H: composition fidelity for every new CIF
    n_read = 0
    for r in v9[v9.origin == "new"].itertuples():
        if r.id not in src.index:
            continue
        p = _os.path.join(_ROOT, src.loc[r.id, "cif"])
        try:
            st = Structure.from_file(p); n_read += 1
        except Exception:  # noqa: BLE001
            flags[r.id].append("H:cif-unreadable"); continue
        g = st.composition.fractional_composition.get_el_amt_dict()
        t = Composition(r.formula).fractional_composition.get_el_amt_dict()
        e = max(abs(g.get(x, 0) - t.get(x, 0)) for x in set(g) | set(t))
        if e > 0.02:
            flags[r.id].append(f"H:comp-err-{e:.3f}")
    # ---- hard / review classification (the rules settled on 2026-09-14) ----
    MIXED = {"Fe", "Co", "Mn", "Cr", "V", "Ni", "Ru", "Ti", "Ce", "Pr", "Tb", "Eu", "Yb", "Sn", "Sb", "Mo", "W",
             "Re", "Os", "Ir", "Rh", "Pd", "Pt", "U"}
    OX2 = dict(OX); OX2.update({"Y": 3, "Bi": 3, "Tl": 3, "Pb": 2, "Hg": 2, "Fe": 3, "Co": 3, "Mn": 3, "Ru": 4, "Ga": 3,
                                "Ti": 4, "Al": 3, "Zn": 2, "Ni": 2, "Cd": 2, "In": 3, "Sb": 3, "Sc": 3, "Lu": 3, "Tb": 3,
                                "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3, "Th": 4, "Ta": 5, "Nb": 5, "Mo": 6, "W": 6,
                                "V": 5, "Cr": 3, "Zr": 4, "Hf": 4, "Mg": 2, "Li": 1, "Rb": 1, "Cs": 1, "Ag": 1, "Au": 3,
                                "Sn": 4, "Ge": 4, "Si": 4, "B": 3, "C": 4, "P": 5, "As": 5, "Re": 7, "Os": 4, "Ir": 4,
                                "Pt": 2, "Pd": 2, "Rh": 3, "U": 4, "Ce": 3, "Tc": 7, "Be": 2, "D": 1})
    hard, review = defaultdict(list), defaultdict(list)
    for r in v9.itertuples():
        c = comps.get(r.id)
        if c is None:
            continue
        rs = flags.get(r.id, [])
        # typo: grossly unphysical Cu oxidation on an essentially pure cuprate
        cations_ = sum(v for e, v in c.items() if e not in ANION)
        if cls(c) == "cuprate" and c["Cu"] >= 0.2 * cations_ and all(e in OX2 for e in c if e != "Cu"):
            cu = c["Cu"]; mixed = sum(v for e, v in c.items() if e in MIXED)
            ox = -sum(OX2[e] * n for e, n in c.items() if e != "Cu") / cu
            if mixed < 0.3 * cu and (ox < 0.8 or ox > 4.0):
                hard[r.id].append(f"typo: Cu oxidation {ox:.2f}")
        for x in rs:
            if x.startswith("G:parent-lacks"):
                hard[r.id].append("structure: parent lacks the redox centre")
            elif x.startswith("G:anion-ratio"):
                dr = float(x.split("anion-ratio-")[1].split("(")[0])
                if abs(dr) > 1.0:
                    hard[r.id].append(f"structure: parent of a different compound class ({dr:+.1f} anions/cation)")
            elif x.startswith("H:comp-err"):
                hard[r.id].append("structure: composition mismatch")
            elif x.startswith("B:alloy") or x.startswith("B:elemental"):
                hard[r.id].append("typo: impossible Tc for chemistry")
            elif x.startswith("A:huge"):
                # real large cells exist (Ba24Si100 clathrate, Al28Ca24O66 mayenite,
                # percent formulas); only a runaway index is a typo
                if max(c.values()) > 500:
                    hard[r.id].append("typo: absurd coefficient")
            elif x.startswith("D:duplicate") and r.tc == 0:
                hi = float(x.split("..")[1].split("K")[0])
                if hi > 20:
                    review[r.id].append(f"label: Tc=0 report at a composition others put at {hi:.0f} K")
        if r.id in src.index and r.origin == "new" and r.id not in hard:
            sr = src.loc[r.id]
            try:
                pc = set(Composition(str(sr.parent_formula)).get_el_amt_dict())
            except Exception:  # noqa: BLE001
                pc = None
            if pc is not None and str(sr.doping).startswith("multi_sublattice"):
                cats = {e: v for e, v in c.items() if e not in ANION}; tot = sum(cats.values())
                new = sum(v for e, v in cats.items() if e not in pc) / tot if tot else 0
                if new > 0.5:      # more foreign than native: the parent's framework is not this compound's
                    hard[r.id].append(f"structure: {100*new:.0f}% of cations foreign to parent {sr.parent_formula}")
                elif new > 0.35:
                    review[r.id].append(f"structure: {100*new:.0f}% of cations foreign to parent {sr.parent_formula}")
    v9i = v9.set_index("id")
    HARD = _os.path.join(_ROOT, "docs", "data_curation", "v9_audit_exclude_ids.csv")
    REV = _os.path.join(_ROOT, "docs", "data_curation", "v9_audit_review_ids.csv")
    pd.DataFrame([dict(id=i, formula=v9i.formula[i], tc=v9i.tc[i], origin=v9i.origin[i], reason="; ".join(w))
                  for i, w in hard.items()]).to_csv(HARD, index=False)
    pd.DataFrame([dict(id=i, formula=v9i.formula[i], tc=v9i.tc[i], origin=v9i.origin[i], reason="; ".join(w))
                  for i, w in review.items()]).to_csv(REV, index=False)
    print(f"HARD exclusions: {len(hard)} (typo {sum(1 for w in hard.values() if any(x.startswith('typo') for x in w))}, "
          f"structure {sum(1 for w in hard.values() if any(x.startswith('structure') for x in w))}) -> {HARD}")
    print(f"REVIEW: {len(review)} -> {REV}")
    rows = []
    for r in v9.itertuples():
        if r.id in flags:
            rs = flags[r.id]
            sev = sum({"A": 1, "B": 3, "C": 2, "D": 2, "E": 1, "F": 3, "G": 2, "H": 3, "I": 3}[x[0]] for x in rs)
            rows.append(dict(id=r.id, formula=r.formula, tc=r.tc, origin=r.origin, severity=sev,
                             n_flags=len(rs), reasons="; ".join(rs)))
    out = pd.DataFrame(rows).sort_values(["severity", "tc"], ascending=False)
    _os.makedirs(_os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"V9 rows {len(v9)} ({(v9.origin=='new').sum()} new) | CIFs read {n_read} | flagged {len(out)} "
          f"({(out.origin=='new').sum()} new, {(out.origin=='V45').sum()} V45) -> {OUT}")
    by = defaultdict(int)
    for rs in flags.values():
        for x in {x.split("(")[0].split("-Tc")[0].split(":")[0] + ":" + x.split(":")[1].split("(")[0].split("-")[0] for x in rs}:
            by[x] += 1
    print("\nflag counts (rows carrying each check):")
    for k_, n in sorted(by.items(), key=lambda x: -x[1]):
        print(f"  {n:5d}  {k_}")


if __name__ == "__main__":
    main()
