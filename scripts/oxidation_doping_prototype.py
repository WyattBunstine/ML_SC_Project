#!/usr/bin/env python3
"""PROTOTYPE + validation: doping-aware oxidation-state assignment.

The current builder guesses oxidation states from the DOPED (fractional) composition,
which pymatgen can't charge-balance -> it defaults doped superconductors to ~0. This
computes them the right way for the SC transfer set:

  RULE (agreed): pin every fixed-valence ion to its definite oxidation state; identify
  the host TRANSITION-METAL redox center; solve that center's oxidation by charge
  balance over the (fractional) doped composition. Reduces to the clean nominal states
  for an undoped/ordered cell; carries the doping-induced charge (hole/electron) onto
  the redox center (the cuprate/pnictide T_c order parameter) for a doped one.

This is composition-level (doped + parent element->amount dicts) — the per-site
decoration is a thin wrapper on top. Run standalone to validate against canonical
superconductors BEFORE wiring into ingestion:

    python scripts/oxidation_doping_prototype.py    # exit 0 = all canonical cases pass
"""
import sys

from pymatgen.core import Composition, Element

# Anions get their definite negative oxidation; everything else is a cation we either
# pin (single definite state) or let float (the redox center).
ANION_OXI = {"O": -2.0, "F": -1.0, "Cl": -1.0, "Br": -1.0, "I": -1.0,
             "S": -2.0, "Se": -2.0, "Te": -2.0, "N": -3.0, "P": -3.0, "As": -3.0}


def _is_redox_candidate(sym: str) -> bool:
    """A d-block transition metal that is actually redox-active: groups 4-11. Excludes
    group 3 (Sc/Y/La/lanthanoids — fixed 3+) and group 12 (Zn/Cd/Hg — fixed 2+), which
    pymatgen also calls 'transition metals' but which never carry the doping charge."""
    el = Element(sym)
    g = el.group
    return el.is_transition_metal and 4 <= g <= 11


def _pinned_cation_oxi(sym: str) -> float:
    """Definite oxidation for a fixed-valence cation. Group 3 + lanthanoids -> +3,
    alkali -> +1, alkaline-earth -> +2; otherwise the element's (single) common state."""
    el = Element(sym)
    if el.is_alkali:
        return 1.0
    if el.is_alkaline:
        return 2.0
    if el.is_lanthanoid or el.group == 3:
        return 3.0
    common = el.common_oxidation_states
    return float(common[0]) if common else 0.0


def _dopant_oxi(sym: str, replaced_host: str | None) -> float:
    """Oxidation of a dopant element. Single common state -> that. Multiple (e.g. Ce
    3/4) -> the one that DIFFERS from the host it replaces, i.e. the aliovalent state
    that actually creates doping (Ce->4 replacing Nd->3 = electron doping)."""
    common = Element(sym).common_oxidation_states
    if not common:
        return _pinned_cation_oxi(sym)
    if len(common) == 1 or replaced_host is None:
        return float(common[0])
    host_oxi = _pinned_cation_oxi(replaced_host)
    aliovalent = [c for c in common if abs(c - host_oxi) > 1e-9 and c > 0]
    return float(aliovalent[0]) if aliovalent else float(common[0])


def assign_oxidation(doped_comp: dict, parent_comp: dict):
    """Per-element oxidation for a doped composition. Returns (element_oxi, info).

    info carries: redox_element, redox_oxi, charge_imbalance (pre-redox), source, and
    `flags` (intermetallic / no_redox_center / multi_redox_candidate / out_of_range)."""
    doped = {str(k): float(v) for k, v in doped_comp.items()}
    parent = {str(k): float(v) for k, v in parent_comp.items()}
    flags = []

    anions = [e for e in doped if e in ANION_OXI]
    if not anions:                                   # all-metal -> oxidation ill-defined
        return ({e: 0.0 for e in doped},
                {"redox_element": None, "source": "intermetallic", "flags": ["intermetallic"]})

    dopants = [e for e in doped if e not in parent]  # elements only in the doped cell
    # which host did each dopant replace? the parent CATION whose amount dropped most.
    host_drops = {e: parent.get(e, 0.0) - doped.get(e, 0.0)
                  for e in parent if e not in ANION_OXI}
    replaced_host = max(host_drops, key=host_drops.get) if host_drops else None

    # redox center: a host group-4..11 TM (prefer 3d = lowest row). Dopant TMs are
    # treated as pinned dopants, not the redox center (e.g. Ce/Co dopants).
    host_redox = [e for e in doped if e not in dopants and _is_redox_candidate(e)]
    host_redox.sort(key=lambda s: (Element(s).row, s))     # 3d before 4d/5d
    if len(host_redox) > 1:
        flags.append(f"multi_redox_candidate:{host_redox}")
    redox = host_redox[0] if host_redox else None

    element_oxi, fixed_charge = {}, 0.0
    for e, amt in doped.items():
        if e == redox:
            continue
        if e in ANION_OXI:
            oxi = ANION_OXI[e]
        elif e in dopants:
            oxi = _dopant_oxi(e, replaced_host)
        else:
            oxi = _pinned_cation_oxi(e)
        element_oxi[e] = oxi
        fixed_charge += amt * oxi

    if redox is None:                                # ionic but no TM to absorb charge
        flags.append("no_redox_center")
        return (element_oxi, {"redox_element": None, "charge_imbalance": fixed_charge,
                              "source": "pinned_only", "flags": flags})

    redox_oxi = -fixed_charge / doped[redox]         # solve neutrality (fractional ok)
    element_oxi[redox] = redox_oxi
    common = Element(redox).common_oxidation_states
    lo, hi = (min(common), max(common)) if common else (0, 6)
    if not (lo - 1.0 <= redox_oxi <= hi + 1.0):      # sanity window around known states
        flags.append(f"out_of_range:{redox}={redox_oxi:.2f}")
    return (element_oxi, {"redox_element": redox, "redox_oxi": redox_oxi,
                          "charge_imbalance": fixed_charge, "source": "redox_balance",
                          "flags": flags})


def site_oxidation_states(structure, parent_comp):
    """Per-site occupancy-weighted oxidation for a (possibly disordered) Structure — the
    decoration ingestion will apply. Element oxidations come from assign_oxidation; each
    site is the occupancy-weighted mix of its species (so a La/Sr mixed site reads
    between La3+ and Sr2+, while the Cu redox site carries the doped charge). Returns
    (list[float] aligned to sites, info)."""
    doped = dict(structure.composition.get_el_amt_dict())
    element_oxi, info = assign_oxidation(doped, parent_comp)
    site_oxi = [sum(occ * element_oxi.get(getattr(sp, "symbol", str(sp)), 0.0)
                    for sp, occ in site.species.items())
                for site in structure]
    return site_oxi, info


def _structural_check():
    """Build a disordered LSCO cell and confirm per-site decoration: the mixed La/Sr
    sites read the occupancy-weighted cation charge, Cu carries the hole, O = -2."""
    from pymatgen.core import Lattice, Structure
    # La1.85Sr0.15CuO4 over the I4/mmm body-centered cell: 2 La/Sr sites (each La .925/Sr .075).
    latt = Lattice.tetragonal(3.78, 13.2)
    species = [{"La": 0.925, "Sr": 0.075}, {"La": 0.925, "Sr": 0.075},
               {"Cu": 1.0}, {"O": 1.0}, {"O": 1.0}, {"O": 1.0}, {"O": 1.0}]
    coords = [[0, 0, 0.36], [0, 0, 0.64], [0, 0, 0], [0, 0, 0.18],
              [0, 0, 0.82], [0.5, 0, 0], [0, 0.5, 0]]
    s = Structure(latt, species, coords)
    site_oxi, info = site_oxidation_states(s, _comp("La2CuO4"))
    by_site = {tuple(sorted(site.species.get_el_amt_dict())): round(o, 3)
               for site, o in zip(s, site_oxi)}
    la_sr = round(0.925 * 3 + 0.075 * 2, 3)
    ok = (abs(by_site[("La", "Sr")] - la_sr) < 1e-3
          and abs(by_site[("Cu",)] - 2.15) < 0.02
          and abs(by_site[("O",)] + 2.0) < 1e-9)
    print(f"\nstructural decoration (LSCO disordered cell): "
          f"La/Sr-site={by_site[('La','Sr')]} (exp {la_sr}), "
          f"Cu={by_site[('Cu',)]} (exp 2.15), O={by_site[('O',)]}  "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------- validation
CASES = [
    # name, doped, parent, expected (element, oxidation) or ("flag", substr)
    ("La2CuO4 (undoped)",   "La2CuO4",            "La2CuO4",      ("Cu", 2.00)),
    ("LSCO x=0.15 (holes)", "La1.85Sr0.15CuO4",   "La2CuO4",      ("Cu", 2.15)),
    ("NCCO x=0.15 (elec.)", "Nd1.85Ce0.15CuO4",   "Nd2CuO4",      ("Cu", 1.85)),
    ("YBCO O7",             "Y1Ba2Cu3O7",         "Y1Ba2Cu3O7",   ("Cu", 2.333)),
    ("YBCO O6.9 (O-vac)",   "Y1Ba2Cu3O6.9",       "Y1Ba2Cu3O7",   ("Cu", 2.267)),
    ("BSCCO-2212",          "Bi2Sr2Ca1Cu2O8",     "Bi2Sr2Ca1Cu2O8", ("Cu", 2.00)),
    ("MgB2 (intermetallic)","Mg1B2",              "Mg1B2",        ("flag", "intermetallic")),
    ("AuGa (intermetallic)","Au0.9Ga0.1",         "Au1Ga1",       ("flag", "intermetallic")),
    ("BaFe2As2 (undoped)",  "Ba1Fe2As2",          "Ba1Fe2As2",    ("Fe", 2.00)),
]


def _comp(formula):
    return dict(Composition(formula).get_el_amt_dict())


def main():
    ok_all = True
    print(f"{'case':24s} {'redox':5s} {'got':>8s} {'expect':>8s}  result   flags")
    for name, doped, parent, expect in CASES:
        oxi, info = assign_oxidation(_comp(doped), _comp(parent))
        flags = ",".join(info.get("flags", [])) or "-"
        if expect[0] == "flag":
            passed = any(expect[1] in f for f in info.get("flags", [])) or info.get("source") == expect[1]
            got = info.get("source", "")
            print(f"{name:24s} {'-':5s} {got:>8s} {expect[1]:>8s}  "
                  f"{'PASS' if passed else 'FAIL':6s}  {flags}")
        else:
            el, exp = expect
            got = oxi.get(el, float('nan'))
            passed = abs(got - exp) < 0.02
            print(f"{name:24s} {el:5s} {got:8.3f} {exp:8.3f}  "
                  f"{'PASS' if passed else 'FAIL':6s}  redox={info.get('redox_element')} {flags}")
        ok_all = ok_all and passed
    ok_all = _structural_check() and ok_all
    print("\noxidation_doping_prototype: " + ("PASS" if ok_all else "FAIL"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
