"""Audit the SC index for spurious doping datapoints — rows whose NOMINAL
composition is physically inconsistent with their Tc label (the unannotated-
oxygen-content class and relatives).

Checks (cuprate/oxide rows only — charge balance means nothing for
intermetallics):
  A  undoped-insulator claiming SC: charge-balanced Cu ox in [1.98, 2.02]
     with Tc > 5 K — a stoichiometric Cu2+ parent cannot superconduct; the
     label belongs to an O-doped/cation-doped sample reported at nominal
     stoichiometry (La2CuO4 "40 K" class).
  B  implausible Cu oxidation with high Tc: ox > 2.6 or < 1.7 while Tc > 10 K
     — nominal compositions implying Cu(III)/Cu(I) regimes where the dome is
     dead (LaSrCuO4 = "Cu+3 at 40 K" class).
  C  dome-violating overdoped: 2.35 < ox <= 2.6 and Tc > 20 K (beyond every
     known single-layer dome edge).
  D  conflicting duplicates: same anonymized composition, Tc spread > 15 K
     across rows (both rows flagged, the disagreement is unresolvable here).
  E  NEMAD source disagreement: nemad_iqr > 10 K (consensus of sources spans
     more than a dome's width).

Writes docs/data_curation/doping_label_suspects.csv with one row per (id,
check) and prints the summary. EXCLUSION IS A PROTOCOL DECISION — feed the csv
(or a filtered subset) to exclude_ids_csv per evaluation round.

Usage: python scripts/audit_doping_labels.py [index.pickle]
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from dome_stats import cu_oxidation  # noqa: E402

IDX = sys.argv[1] if len(sys.argv) > 1 else "database/datafiles/MP/SC_MP_V6_v45.pickle"
idx = pd.read_pickle(IDX)
idx["formula"] = idx.id.map(lambda i: re.split(r"-MP-|-ICSD-|-NEMAD", str(i))[0])

# Extended fixed-valence map for the charge balance. dome_stats' map only
# covers the LSCO family; here every non-Cu element must be covered or the row
# is UNASSESSABLE (skipped, never flagged) — Bi/Tl/Pb are mixed-valence in
# real samples, so their families are deliberately unassessable rather than
# flagged with a pretend charge balance.
OX_FULL = {"La": 3, "Sr": 2, "Ba": 2, "Ce": 4, "Nd": 3, "Eu": 3, "Gd": 3,
           "Sm": 3, "Pr": 3, "Ca": 2, "K": 1, "Na": 1, "Li": 1, "Rb": 1,
           "Cs": 1, "Y": 3, "Dy": 3, "Ho": 3, "Er": 3, "Tm": 3, "Yb": 3,
           "Lu": 3, "Tb": 3, "Sc": 3, "Mg": 2, "Zn": 2, "Cd": 2, "Hg": 2,
           "O": -2, "F": -1, "Cl": -1, "Br": -1}


def cu_ox_strict(formula):
    try:
        from pymatgen.core import Composition
        c = Composition(formula).get_el_amt_dict()
        cu = c.get("Cu", 0)
        if cu <= 0:
            return None
        if any(e not in OX_FULL for e in c if e != "Cu"):
            return None            # unassessable (mixed-valence cations etc.)
        return -sum(OX_FULL[e] * n for e, n in c.items() if e != "Cu") / cu
    except Exception:
        return None


idx["cu_ox"] = idx.formula.map(cu_ox_strict)
is_cu_oxide = idx.cu_ox.notna() & idx.formula.str.contains("O")

sus = []


def flag(mask, check, why):
    for r in idx[mask].itertuples():
        sus.append({"id": r.id, "check": check, "tc": r.tc,
                    "cu_ox": round(r.cu_ox, 3) if pd.notna(r.cu_ox) else "",
                    "reason": why})
    print(f"  {check}: {int(mask.sum())} rows")


print(f"auditing {IDX} ({len(idx)} rows; {int(is_cu_oxide.sum())} Cu-oxide)")
flag(is_cu_oxide & idx.cu_ox.between(1.98, 2.02) & (idx.tc > 5),
     "A_undoped_insulator_sc", "charge-balanced Cu2+ (undoped) with Tc>5K — label from O/cation-doped sample")
flag(is_cu_oxide & ((idx.cu_ox > 2.6) | (idx.cu_ox < 1.7)) & (idx.tc > 10),
     "B_implausible_cu_ox", "nominal Cu ox outside [1.7,2.6] with Tc>10K — O-content artifact")
flag(is_cu_oxide & idx.cu_ox.between(2.35, 2.6, inclusive="neither") & (idx.tc > 20),
     "C_overdoped_dome_violation", "Cu ox 2.35-2.6 with Tc>20K — beyond known dome edges")

# D: conflicting duplicates on normalized composition
def normf(f):
    try:
        from pymatgen.core import Composition
        return Composition(f).fractional_composition.alphabetical_formula
    except Exception:
        return f
idx["nform"] = idx.formula.map(normf)
g = idx.groupby("nform").tc.agg(["min", "max", "count"])
dup = set(g[(g["count"] > 1) & (g["max"] - g["min"] > 15)].index)
flag(idx.nform.isin(dup), "D_conflicting_duplicates",
     "same composition, Tc spread >15K across rows")

# E: NEMAD internal disagreement
if "nemad_iqr" in idx.columns:
    flag(idx.nemad_iqr.fillna(0) > 10, "E_nemad_iqr",
         "NEMAD source IQR >10K — sources disagree by more than a dome width")

out = pd.DataFrame(sus)
os.makedirs("docs/data_curation", exist_ok=True)
out.to_csv("docs/data_curation/doping_label_suspects.csv", index=False)
uniq = out.id.nunique() if len(out) else 0
multi = (out.groupby("id").size() > 1).sum() if len(out) else 0
print(f"\n{uniq} unique suspect rows ({multi} flagged by 2+ checks) "
      f"-> docs/data_curation/doping_label_suspects.csv")
if len(out):
    ex = out[out.check.isin(["A_undoped_insulator_sc", "B_implausible_cu_ox"])][["id"]].drop_duplicates()
    ex.to_csv("docs/data_curation/exclude_charge_impossible.csv", index=False)
    print(f"proposed hard-exclusion list (checks A+B only): {len(ex)} rows "
          f"-> docs/data_curation/exclude_charge_impossible.csv")
