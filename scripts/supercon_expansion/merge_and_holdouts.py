"""Merge the SuperCon expansion into a training index (V8) and define the
Matthias valence-dome holdouts.

V8 = SC_MP_V4_doped_v45 (5,773) + SC_MP_V7_supercon (7,896), de-duplicated on
the normalized fractional composition (V45 wins). Descriptors and the 3DSC
family metadata are extended in step so no row is dropped or lands in "Other"
by accident. Holdouts hold out whole ALLOY SYSTEMS (element pairs/triples),
never single compositions, so a held-out series is genuinely unseen.
  python scripts/supercon_expansion/merge_and_holdouts.py
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_HERE))
_sys.path.insert(0, _ROOT)
import warnings  # noqa: E402
import numpy as np, pandas as pd  # noqa: E402
warnings.filterwarnings("ignore")
from pymatgen.core import Composition  # noqa: E402

MP = _os.path.join(_ROOT, "database", "datafiles", "MP")
VAL = {"Sc": 3, "Y": 3, "La": 3, "Lu": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
       "Cr": 6, "Mo": 6, "W": 6, "Mn": 7, "Tc": 7, "Re": 7, "Fe": 8, "Ru": 8, "Os": 8,
       "Co": 9, "Rh": 9, "Ir": 9, "Ni": 10, "Pd": 10, "Pt": 10}
# The two Matthias humps, by ALLOY SYSTEM. Membership is by element set, so every
# composition of a held-out system leaves training together.
DOME1 = ["Nb-Ti", "Ta-Ti", "V-Zr", "Ti-Zr", "Mo-Zr", "Hf-Mo", "Hf-W", "Nb-Zr", "Ti-V",
         "Mo-Ti", "Ta-Zr", "Nb-V", "Hf-Nb", "Hf-Ti", "Hf-V", "Nb-Ti-Zr", "Nb-V-Zr",
         "Nb-Ta-Ti", "Nb-Ti-V", "Ti-V-Zr"]
DOME2 = ["Mo-Re", "Mo-Tc", "Re-W", "Mo-Ru", "Os-Re", "Nb-Re", "Re-Ta", "Mo-Nb", "Cr-Re",
         "Mo-W", "Re-V", "Mo-Os", "Mo-Pt-Re", "Re-Ru", "Cr-Ru", "Mo-Re-Ru"]


def key(f):
    try:
        c = Composition(f).fractional_composition.get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return None
    return "|".join(f"{e}{round(v, 3)}" for e, v in sorted(c.items()))


def formula(i):
    return str(i).split("-MP-")[0].split("-ICSD-")[0]


def ea(f):
    try:
        c = Composition(f).get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return None
    return sum(VAL[e] * a for e, a in c.items()) / sum(c.values()) if set(c) <= set(VAL) else None


def system(f):
    try:
        return "-".join(sorted(Composition(f).get_el_amt_dict()))
    except Exception:  # noqa: BLE001
        return None


def fam(f):
    try:
        c = set(Composition(f).get_el_amt_dict())
    except Exception:  # noqa: BLE001
        return "Other"
    if "Cu" in c and "O" in c:
        return "Cuprate"
    if "Fe" in c and c & {"As", "P", "Se", "Te", "S"}:
        return "Ferrite"
    if "O" in c:
        return "Oxide"
    if c & {"Ce", "U", "Yb", "Np", "Pu"} and len(c) > 1:
        return "Heavy_fermion"
    return "Other"


def main():
    v45 = pd.read_pickle(_os.path.join(MP, "SC_MP_V4_doped_v45.pickle"))
    v7 = pd.read_pickle(_os.path.join(MP, "SC_MP_V7_supercon.pickle"))
    v7["graph_path"] = v7.graph_path.map(lambda p: _os.path.relpath(p, _ROOT) if _os.path.isabs(p) else p)
    v45["key"] = v45.id.map(lambda i: key(formula(i)))
    v7["key"] = v7.id.map(lambda i: key(formula(i)))
    dup = v7.key.isin(set(v45.key.dropna()))
    print(f"V45 {len(v45)} + V7 {len(v7)} | V7 rows duplicating a V45 composition: {int(dup.sum())} (dropped)")
    v8 = pd.concat([v45, v7[~dup]], ignore_index=True)
    v8 = v8.drop_duplicates("id").drop(columns=["key"])
    out = _os.path.join(MP, "SC_MP_V8_supercon.pickle")
    v8.to_pickle(out)
    v8.to_csv(out.replace(".pickle", ".csv"), index=False)
    print(f"V8 index: {len(v8)} rows -> {out}")
    # ---- descriptors ----
    d45 = pd.read_pickle(_os.path.join(MP, "descriptors_doped.pickle"))
    d7 = pd.read_pickle(_os.path.join(MP, "descriptors_v7_supercon.pickle"))
    assert list(d45["names"]) == list(d7["names"]), "descriptor layouts differ"
    table = dict(d45["table"]); table.update(d7["table"])
    miss = [i for i in v8.id if i not in table]
    pd.to_pickle({"names": list(d45["names"]), "table": table, "failed": []},
                 _os.path.join(MP, "descriptors_v8_supercon.pickle"))
    print(f"descriptors: {len(table)} rows ({len(miss)} index rows missing -> would be DROPPED)")
    # ---- family metadata: append V7 ids so they aren't silently 'Other' ----
    meta = pd.read_csv(_os.path.join(MP, "3DSC_MP.csv"), low_memory=False)
    have = set(meta.cif.astype(str).map(_os.path.basename))
    new = [{"cif": i, "sc_class": fam(formula(i)), "synth_doped": True}
           for i in v8.id if i not in have]
    meta2 = pd.concat([meta, pd.DataFrame(new)], ignore_index=True)
    meta2.to_csv(_os.path.join(MP, "3DSC_MP_v8.csv"), index=False)
    print(f"metadata: +{len(new)} rows -> 3DSC_MP_v8.csv "
          f"({pd.Series([r['sc_class'] for r in new]).value_counts().to_dict()})")
    # ---- Matthias holdouts ----
    v8["formula"] = v8.id.map(formula)
    v8["sys"] = v8.formula.map(system)
    v8["ea"] = v8.formula.map(ea)
    alloy = v8[v8.ea.notna()]
    print(f"\nall-transition-metal alloy rows in V8: {len(alloy)} across {alloy.sys.nunique()} systems")
    for tag, systems in (("dome1", DOME1), ("dome2", DOME2), ("all", sorted(alloy.sys.unique()))):
        sel = alloy[alloy.sys.isin(systems)]
        p = _os.path.join(MP, f"holdout_matthias_{tag}_ids.csv")
        pd.DataFrame({"id": sel.id}).to_csv(p, index=False)
        pos = sel[sel.tc > 0]
        print(f"  {tag:6s} {len(sel):4d} rows ({len(pos)} with Tc>0) over {sel.sys.nunique():3d} systems, "
              f"e/a {sel.ea.min():.2f}-{sel.ea.max():.2f}, Tc<=" f"{sel.tc.max():.1f} K -> {_os.path.basename(p)}")


if __name__ == "__main__":
    main()
