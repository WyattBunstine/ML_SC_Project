#!/usr/bin/env python3
"""Permanent gate for the doping-aware oxidation-state logic in
database/oxidation_doping.py — validates the redox rule against canonical
superconductors (the chemistry that must stay correct as the rule evolves).

    python scripts/oxidation_doping_prototype.py    # exit 0 = all canonical cases pass

The full-3DSC flag audit is a separate one-off (see the commit history); this is the
fast must-pass chemistry gate, imported by nothing else so it never drifts from the module.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.oxidation_doping import assign_oxidation, site_oxidation_states  # noqa: E402


CASES = [
    # name, doped, parent, expected (element, oxidation) or ("flag", source/substr)
    ("La2CuO4 (undoped)",     "La2CuO4",            "La2CuO4",        ("Cu", 2.00)),
    ("LSCO x=0.15 (holes)",   "La1.85Sr0.15CuO4",   "La2CuO4",        ("Cu", 2.15)),
    ("NCCO x=0.15 (elec.)",   "Nd1.85Ce0.15CuO4",   "Nd2CuO4",        ("Cu", 1.85)),
    ("YBCO O7",               "Y1Ba2Cu3O7",         "Y1Ba2Cu3O7",     ("Cu", 2.333)),
    ("YBCO O6.9 (O-vac)",     "Y1Ba2Cu3O6.9",       "Y1Ba2Cu3O7",     ("Cu", 2.267)),
    ("BSCCO-2212",            "Bi2Sr2Ca1Cu2O8",     "Bi2Sr2Ca1Cu2O8", ("Cu", 2.00)),
    ("MgB2 (intermetallic)",  "Mg1B2",              "Mg1B2",          ("flag", "intermetallic")),
    ("AuGa (intermetallic)",  "Au0.9Ga0.1",         "Au1Ga1",         ("flag", "intermetallic")),
    ("BaFe2As2 (undoped)",    "Ba1Fe2As2",          "Ba1Fe2As2",      ("Fe", 2.00)),
]


def _comp(formula):
    from pymatgen.core import Composition
    return dict(Composition(formula).get_el_amt_dict())


def _structural_check():
    """Per-site decoration on a disordered LSCO cell: mixed La/Sr sites read the
    occupancy-weighted cation charge, Cu carries the hole, O = -2."""
    from pymatgen.core import Lattice, Structure
    latt = Lattice.tetragonal(3.78, 13.2)
    species = [{"La": 0.925, "Sr": 0.075}, {"La": 0.925, "Sr": 0.075},
               {"Cu": 1.0}, {"O": 1.0}, {"O": 1.0}, {"O": 1.0}, {"O": 1.0}]
    coords = [[0, 0, 0.36], [0, 0, 0.64], [0, 0, 0], [0, 0, 0.18],
              [0, 0, 0.82], [0.5, 0, 0], [0, 0.5, 0]]
    s = Structure(latt, species, coords)
    site_oxi, _ = site_oxidation_states(s, _comp("La2CuO4"))
    by_site = {tuple(sorted(site.species.get_el_amt_dict())): round(o, 3)
               for site, o in zip(s, site_oxi)}
    la_sr = round(0.925 * 3 + 0.075 * 2, 3)
    ok = (abs(by_site[("La", "Sr")] - la_sr) < 1e-3
          and abs(by_site[("Cu",)] - 2.15) < 0.02 and abs(by_site[("O",)] + 2.0) < 1e-9)
    print(f"\nstructural decoration (LSCO disordered cell): La/Sr-site={by_site[('La','Sr')]} "
          f"(exp {la_sr}), Cu={by_site[('Cu',)]} (exp 2.15), O={by_site[('O',)]} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    warnings.simplefilter("ignore")
    ok_all = True
    print(f"{'case':24s} {'redox':5s} {'got':>8s} {'expect':>8s}  result   flags")
    for name, doped, parent, expect in CASES:
        oxi, info = assign_oxidation(_comp(doped), _comp(parent))
        flags = ",".join(info.get("flags", [])) or "-"
        if expect[0] == "flag":
            passed = info.get("source") == expect[1] or any(expect[1] in f for f in info.get("flags", []))
            print(f"{name:24s} {'-':5s} {info.get('source',''):>8s} {expect[1]:>8s}  "
                  f"{'PASS' if passed else 'FAIL':6s}  {flags}")
        else:
            el, exp = expect
            got = oxi.get(el, float('nan'))
            passed = abs(got - exp) < 0.02
            print(f"{name:24s} {el:5s} {got:8.3f} {exp:8.3f}  "
                  f"{'PASS' if passed else 'FAIL':6s}  redox={info.get('redox_element')} {flags}")
        ok_all = ok_all and passed
    ok_all = _structural_check() and ok_all
    print("\noxidation_doping gate: " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
