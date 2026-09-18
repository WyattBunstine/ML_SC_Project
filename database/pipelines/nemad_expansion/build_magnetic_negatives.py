"""Phase 1 of the magnetic-negatives plan: consensus + SC-overlap filter.

Turns NEMAD's magnetic_materials.csv (33.7k reports: Curie/Neel temps per
composition) into a candidate list of tc=0 HARD NEGATIVES for the T_c head —
materials whose ground state is magnetic order, not superconductivity. These
target the observed failure mode (magnetic perovskite parents predicted SC):
nothing in the current training data says "this composition orders instead".

Overlap policy is FAMILY-AWARE: a candidate is dropped only when ITS OWN
composition matches an SC entry with tc>0 at the matcher's identical/similar
tiers (coexistence or report disagreement -> ambiguous). Compositions at mere
DOPING distance from an SC entry (the undoped magnetic parents: La2CuO4-AFM,
LaNiO3, ...) are KEPT — they are the highest-value negatives, and parent_comp
grouping keeps them fold-consistent with their SC families downstream.

Ordering type (FM from Curie, AFM from Neel) and T_order are carried through
for the phase-2 ground-state classifier.

Output: database/datafiles/NE_SCDB/magnetic_negatives.csv
"""
import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _HERE)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_HERE))))

import re
from collections import defaultdict

import numpy as np
import pandas as pd
from pymatgen.core import Composition

from formula_match import chem_dict  # the shared 3DSC totreldiff matcher module


def parse_temp(*vals):
    """First parseable temperature (K) among report strings ('585 °C', '42±1 K',
    '43-62 Oe' -> reject via range guard, plain '50'). Celsius converted."""
    for v in vals:
        if pd.isna(v):
            continue
        s = str(v)
        nums = re.findall(r"[-+]?\d+\.?\d*", s.replace("±", " "))
        if not nums:
            continue
        t = float(nums[0])
        if "°C" in s or "℃" in s:
            t += 273.15
        if 0 < t < 2000:
            return t
    return np.nan


def comp_key(*cands):
    """Canonical fractional-composition key (matches the SC pipeline's keying)."""
    for cand in cands:
        if pd.isna(cand):
            continue
        try:
            d = Composition(str(cand)).get_el_amt_dict()
            if d and all(v > 0 for v in d.values()):
                return "|".join(f"{e}{round(v, 3)}" for e, v in sorted(d.items()))
        except Exception:  # noqa: BLE001
            continue
    return None


def main():
    m = pd.read_csv("database/datafiles/NE_SCDB/magnetic_materials.csv", low_memory=False)
    m["T_curie"] = [parse_temp(a, b) for a, b in zip(m["Curie"], m["Curie(Tc)"])]
    m["T_neel"] = [parse_temp(a, b) for a, b in zip(m["Neel"], m["Neel(Tn)"])]
    m["order"] = np.where(m.T_curie.notna() & m.T_neel.notna(), "both",
                  np.where(m.T_curie.notna(), "FM",
                  np.where(m.T_neel.notna(), "AFM", "none")))
    m = m[m.order != "none"].copy()
    m["key"] = [comp_key(a, b) for a, b in zip(m["New_Column_Concatenated"], m["Material_Name"])]
    m = m[m.key.notna()]
    print(f"reports with temp+composition: {len(m)}")

    # Consensus per composition: median ordering temp, majority ordering type,
    # report count as a confidence weight (mirrors the SC pipeline's weight col).
    m["T_order"] = m[["T_curie", "T_neel"]].max(axis=1)
    cons = (m.groupby("key")
             .agg(formula=("Material_Name", "first"),
                  T_order=("T_order", "median"),
                  order=("order", lambda s: s.mode()[0]),
                  n_reports=("key", "size"))
             .reset_index())
    print(f"unique magnetic compositions: {len(cons)} "
          f"({(cons.order == 'FM').sum()} FM, {(cons.order == 'AFM').sum()} AFM, "
          f"{(cons.order == 'both').sum()} both)")

    # ---- SC references with tc>0: NEMAD-SC consensus + 3DSC ----
    sc_cd = []
    nem = pd.read_csv("database/datafiles/NE_SCDB/nemad_candidates.csv")
    for r in nem.itertuples():
        if r.tc and r.tc > 0:
            try:
                sc_cd.append(chem_dict(r.formula))
            except Exception:  # noqa: BLE001
                pass
    tdsc = pd.read_csv("database/datafiles/MP/3DSC_MP.csv", low_memory=False)
    for r in tdsc.itertuples():
        if r.tc and r.tc > 0:
            try:
                sc_cd.append(chem_dict(r.formula_sc))
            except Exception:  # noqa: BLE001
                pass
    by_sys = defaultdict(list)
    for cd in sc_cd:
        by_sys[frozenset(cd)].append(cd)
    print(f"SC references (tc>0): {len(sc_cd)} across {len(by_sys)} element systems")

    # ---- overlap filter: drop ONLY same-composition matches to a tc>0 SC entry —
    # per-element resolution, NOT the matcher's similar-tier: oxygen-interstitial
    # doping is a tiny totreldiff (La2CuO4 vs SC La2CuO4.086 -> trd~0.012, "similar"),
    # but O-doping IS the doping axis for these families, so the undoped AFM parent
    # is a genuinely different composition and must be KEPT as the hard negative.
    # Drop iff every element agrees within noise (|delta| <= 0.03 per f.u. after
    # scale normalization) — i.e. the magnetic report is of the SC composition itself.
    def is_sc_overlap(key):
        cd = {e_v.rstrip("0123456789."): float(re.sub(r"^[A-Za-z]+", "", e_v))
              for e_v in key.split("|")}
        els = sorted(cd)
        q2 = np.array([cd[e] for e in els])
        for sc in by_sys.get(frozenset(cd), ()):
            q1 = np.array([sc[e] for e in els])
            q2n = q2 * (q1.sum() / q2.sum())
            if np.all(np.abs(q1 - q2n) <= 0.03):
                return True
        return False

    cons["sc_overlap"] = cons.key.map(is_sc_overlap)
    kept = cons[~cons.sc_overlap].drop(columns=["sc_overlap"])
    print(f"dropped as SC-overlapping (identical/similar tier, tc>0): "
          f"{int(cons.sc_overlap.sum())}; kept: {len(kept)}")
    out = "database/datafiles/NE_SCDB/magnetic_negatives.csv"
    kept.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
