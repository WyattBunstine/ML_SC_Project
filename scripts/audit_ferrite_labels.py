"""Iron-based (sc_class Ferrite) label audit — the family-specific analog of
audit_doping_labels.py's cuprate checks (charge balance means nothing for
pnictides, and the corpus-wide near-duplicate threshold of 30 K was tuned to
cuprate Tc scales: it flagged zero ferrites).

Frames by composition: 1111 LnFeAs(P)O / 122 AeFe2As(P)2 / 111 (Li,Na)FeAs /
11 Fe(Se,Te,S) / 245-intercalate AFe2Se2. Checks:
  FA  stoichiometric magnetic parent claiming SC: exact LnFeAsO or AeFe2As2
      with Tc > 5 K — undoped 1111/122 parents are SDW metals; the label
      belongs to a doped/O-deficient sample (111/11 excluded: LiFeAs and FeSe
      superconduct stoichiometrically).
  FB  above the sub-family bulk ceiling: 122 > 42 K, plain 11 > 17 K
      (monolayer/pressure labels), 111 > 27 K, 245 > 34 K.
  FF  near-duplicate conflict: same MP parent + element set, composition
      within 1%, Tc apart by > 12 K (family-scaled version of check F).
  FG  optimal-doping zero: canonical SC windows labeled Tc = 0 — 1111 with
      F 0.08-0.30, 122 with K 0.25-0.55 or Co 0.05-0.15 (per Fe2), 11 with
      Te 0.35-0.65 — absent a secondary heavy substitution.

Same output convention: appends docs/data_curation/ferrite_label_suspects.csv;
EXCLUSION IS A PROTOCOL DECISION (adjudicate series-by-series first).

Usage: python scripts/audit_ferrite_labels.py [index.pickle]
"""
import os
import re
import sys

import pandas as pd
from pymatgen.core import Composition

IDX = sys.argv[1] if len(sys.argv) > 1 else "database/datafiles/MP/SC_MP_V4_doped_v45.pickle"
idx = pd.read_pickle(IDX)
meta = pd.read_csv("database/datafiles/MP/3DSC_MP.csv")
fam = dict(zip(meta["formula_sc"], meta["sc_class"]))
idx["formula"] = idx.id.map(lambda i: re.split(r"-MP-|-ICSD-|-NEMAD", str(i))[0])
idx = idx[idx["formula"].map(lambda f: fam.get(f) == "Ferrite")].copy()

LN = {"La", "Ce", "Pr", "Nd", "Sm", "Gd", "Tb", "Dy", "Y", "Eu", "Ho", "Er"}
AE122 = {"Ba", "Sr", "Ca", "Eu"}


def comp(f):
    try:
        return Composition(f).get_el_amt_dict()
    except Exception:
        return None


idx["c"] = idx["formula"].map(comp)
idx = idx[idx["c"].notna()]


def frame(c):
    els = set(c)
    if c.get("Fe", 0) == 0:
        return "no-Fe"
    has_pn = "As" in els or "P" in els
    chal = els & {"Se", "Te", "S"}
    if has_pn and "O" in els and els & LN:
        return "1111"
    if has_pn and "O" not in els and els & AE122:
        return "122"
    if has_pn and els & {"Li", "Na"} and "O" not in els:
        return "111"
    if chal and not has_pn and not (els & {"K", "Rb", "Cs", "Tl", "Li"}) and "O" not in els:
        return "11"
    if chal and els & {"K", "Rb", "Cs", "Tl"}:
        return "245"
    return "other"


idx["frame"] = idx["c"].map(frame)
sus = []


def flag(mask, check, why):
    for r in idx[mask].itertuples():
        sus.append({"id": r.id, "check": check, "tc": r.tc, "frame": r.frame, "reason": why})
    print(f"  {check}: {int(mask.sum())} rows")


def is_exact_parent(c, fr):
    """Integer-stoichiometric magnetic parent (no fractional site, no dopant)."""
    if fr == "1111":
        want = {"Fe": 1, "As": 1, "O": 1}
        ln = [e for e in c if e in LN]
        return (len(ln) == 1 and c.get(ln[0]) == 1
                and all(c.get(k) == v for k, v in want.items()) and len(c) == 4)
    if fr == "122":
        ae = [e for e in c if e in AE122]
        return (len(ae) == 1 and c.get(ae[0]) == 1 and c.get("Fe") == 2
                and (c.get("As", 0) == 2 or c.get("P", 0) == 2) and len(c) == 3)
    return False


flag(idx.apply(lambda r: is_exact_parent(r["c"], r["frame"]), axis=1) & (idx.tc > 5),
     "FA_parent_claiming_sc",
     "stoichiometric SDW parent with Tc>5K — label from doped/O-deficient sample")

CEIL = {"122": 42.0, "11": 17.0, "111": 27.0, "245": 34.0}
flag(idx.apply(lambda r: r["tc"] > CEIL.get(r["frame"], 1e9), axis=1),
     "FB_above_frame_ceiling",
     "Tc above the sub-family bulk ceiling — film/pressure/intercalate label")

# FF: family-scaled near-duplicate conflicts (same parent + element set, <1%, >12K)
idx["parent"] = idx.id.map(lambda i: (str(i).split("-MP-")[1].split("-synth")[0]
                                      if "-MP-" in str(i) else str(i)))
def _fc(c):
    tot = sum(c.values())
    return {e: v / tot for e, v in c.items()}
idx["fc"] = idx["c"].map(_fc)
near = pd.Series(False, index=idx.index)
for (_els, _par), grp in idx.groupby([idx["c"].map(lambda c: frozenset(c)), "parent"]):
    rows = list(grp.index)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = idx.fc[rows[i]], idx.fc[rows[j]]
            if (sum(abs(a[e] - b[e]) for e in a) < 0.01
                    and abs(idx.tc[rows[i]] - idx.tc[rows[j]]) > 12):
                near[rows[i]] = near[rows[j]] = True
flag(near, "FF_near_duplicate_conflict",
     "composition within 1% of a row whose Tc differs by >12K — twin-report conflict")


def in_sc_window(c, fr):
    heavy = sum(c.get(e, 0) for e in ("Co", "Ni", "Mn", "Cr", "Cu", "Zn", "Ru", "Ir", "Rh"))
    if fr == "1111" and heavy < 0.05:
        return 0.08 <= c.get("F", 0) / max(c.get("O", 0) + c.get("F", 0), 1e-9) <= 0.30
    if fr == "122":
        fe = max(c.get("Fe", 0), 1e-9)
        k_frac = c.get("K", 0) / max(sum(c.get(a, 0) for a in AE122) + c.get("K", 0), 1e-9)
        co_per2 = c.get("Co", 0) * 2.0 / (fe + c.get("Co", 0) + c.get("Ni", 0))
        return (0.25 <= k_frac <= 0.55 and heavy < 0.05) or (0.05 <= co_per2 <= 0.15)
    if fr == "11" and heavy < 0.05:
        se_te = c.get("Se", 0) + c.get("Te", 0)
        return se_te > 0 and 0.35 <= c.get("Te", 0) / se_te <= 0.65
    return False


flag(idx.apply(lambda r: in_sc_window(r["c"], r["frame"]), axis=1) & (idx.tc == 0),
     "FG_optimal_window_zero",
     "canonical SC doping window labeled Tc=0 — sample/label artifact candidate")

out = pd.DataFrame(sus)
os.makedirs("docs/data_curation", exist_ok=True)
out.to_csv("docs/data_curation/ferrite_label_suspects.csv", index=False)
print(f"\n{out.id.nunique() if len(out) else 0} unique ferrite suspects "
      f"-> docs/data_curation/ferrite_label_suspects.csv")
