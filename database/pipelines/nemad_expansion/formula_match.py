"""The 3DSC totreldiff formula matcher — SINGLE shared implementation.

Faithful re-implementation of 3DSC's similarity criterion (totreldiff =
2*sum|d|/(sum n1 + sum n2) with per-element guards), previously duplicated in
match_nemad2.py and match_and_dope.py (flagged in the 2026-07-14 review): any
threshold tweak applied to one copy silently made pipeline stages disagree.
Import from here; no side effects.
"""
import numpy as np
from pymatgen.core import Composition

# (max-reldiff, totreldiff, max-absdiff) guards for the LOW (similar) and HIGH
# (doped-variant) tiers — values from 3DSC's run_pipeline.
LMR, LTR, LMA = 0.10001, 0.05001, 0.15001
HMR, HTR, HMA = 0.20001, 0.15001, 0.3001


def chem_dict(comp):
    """Element -> amount dict, oxygen listed last (3DSC convention)."""
    d = Composition(comp).get_el_amt_dict() if isinstance(comp, str) else dict(comp)
    els = sorted([e for e in d if e != "O"]) + (["O"] if "O" in d else [])
    return {e: float(d[e]) for e in els}


def formula_similarity(cd_sc, cd_2):
    """(tier, totreldiff) of candidate cd_2 against reference cd_sc.
    tier 1 = identical, 2 = similar (low thresholds), 3 = doped variant (high),
    NaN = no match. Requires cd_sc's element set to contain cd_2's."""
    els_sc, els_2 = list(cd_sc), list(cd_2)
    if len(els_sc) < len(els_2) or not all(e in els_sc for e in els_2):
        return np.nan, np.nan
    if len(els_sc) > len(els_2) and len(els_2) == 1:
        return np.nan, np.nan
    q_sc = np.array([cd_sc[e] for e in els_sc])
    q_2 = np.array([cd_2.get(e, 0.0) for e in els_sc])
    norm = q_sc.sum() / q_2.sum()
    q_2 = q_2 * norm
    diffs = np.abs(q_sc - q_2)
    reldiffs = 2 * diffs / (q_sc + q_2)
    trd = 2 * diffs.sum() / (q_sc.sum() + q_2.sum())

    def ok(mr, tr, ma):
        if trd > tr:
            return False
        return not any((d > ma and rd > mr) for d, rd in zip(diffs, reldiffs))

    if trd == 0:
        return 1, trd
    if ok(LMR, LTR, LMA):
        return 2, trd
    if ok(HMR, HTR, HMA):
        return 3, trd
    return np.nan, trd
