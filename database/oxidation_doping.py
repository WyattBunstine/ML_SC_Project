"""Doping-aware oxidation-state assignment for the superconductor (3DSC) transfer set.

pymatgen's `oxi_state_guesses` can't charge-balance a fractional (doped) composition, so
the graph builder defaults doped superconductors to oxidation ~0 — nulling the one channel
that should carry the charge-doping signal (the cuprate/pnictide T_c order parameter).

This computes oxidation states the right way and decorates the structure so the builder's
existing explicit-oxidation path uses them (occupancy-weighted per site):

  RULE: pin every fixed-valence ion to its definite oxidation; identify the host
  transition-metal redox center (group 4-11, coinage metals deprioritized); solve that
  center's oxidation by charge balance over the fractional doped composition. Reduces to
  clean nominal states for an ordered/undoped cell; carries the doping-induced hole/electron
  onto the redox center for a doped one. Implausible solves are pinned (never garbage).

Validated on canonical superconductors + audited across the full 3DSC set — see
scripts/oxidation_doping_prototype.py (the permanent gate, which imports this module).
"""
from pymatgen.core import Composition, Element

# Anions get their definite negative oxidation; everything else is a cation we either pin
# (single definite state) or let float (the redox center).
ANION_OXI = {"O": -2.0, "F": -1.0, "Cl": -1.0, "Br": -1.0, "I": -1.0,
             "S": -2.0, "Se": -2.0, "Te": -2.0, "N": -3.0, "P": -3.0, "As": -3.0}

# Coinage metals are group-11 like Cu but resist high oxidation (Ag/Au are usually +1/+3
# in chalcogenides), so they make poor redox centers — float them only as a last resort.
# The 3DSC audit traced the worst out-of-range solves (Ag=11) to floating Ag.
COINAGE = {"Ag", "Au"}


def _is_redox_candidate(sym: str) -> bool:
    """A d-block transition metal that is actually redox-active: groups 4-11. Excludes
    group 3 (Sc/Y/La/lanthanoids — fixed 3+) and group 12 (Zn/Cd/Hg — fixed 2+), which
    pymatgen also calls 'transition metals' but which never carry the doping charge."""
    el = Element(sym)
    return el.is_transition_metal and 4 <= el.group <= 11


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
    """Oxidation of a dopant element. Single common state -> that. Multiple (e.g. Ce 3/4)
    -> the aliovalent one that DIFFERS from the host it replaces, i.e. the state that
    actually creates doping (Ce->4 replacing Nd->3 = electron doping)."""
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

    info: redox_element, redox_oxi, charge_imbalance, source, and `flags`
    (intermetallic / no_redox_center / multi_redox_candidate / out_of_range_pinned)."""
    doped = {str(k): float(v) for k, v in doped_comp.items()}
    parent = {str(k): float(v) for k, v in parent_comp.items()}
    flags = []

    anions = [e for e in doped if e in ANION_OXI]
    if not anions:                                   # all-metal -> oxidation ill-defined
        return ({e: 0.0 for e in doped},
                {"redox_element": None, "source": "intermetallic", "flags": ["intermetallic"]})

    dopants = [e for e in doped if e not in parent]  # elements only in the doped cell
    host_drops = {e: parent.get(e, 0.0) - doped.get(e, 0.0)
                  for e in parent if e not in ANION_OXI}
    replaced_host = max(host_drops, key=host_drops.get) if host_drops else None

    host_redox = [e for e in doped if e not in dopants and _is_redox_candidate(e)]
    # Priority: non-coinage before Ag/Au; then most ABUNDANT host TM (the majority TM
    # forms the framework — Cu over Ti, majority Fe over minority Co, Mo over Ag in
    # Chevrel); then higher group (later 3d = more redox-active); then symbol.
    host_redox.sort(key=lambda s: (s in COINAGE, -doped[s], -Element(s).group, s))
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
    common = Element(redox).common_oxidation_states
    lo, hi = (min(common), max(common)) if common else (0, 6)
    if not (lo - 1.0 <= redox_oxi <= hi + 1.0):
        # Implausible solve (false redox center, or a tiny redox amount dividing a large
        # charge): DON'T emit a garbage oxidation — pin the center to its common state and
        # accept a residual imbalance, flagged. Keeps the channel sane on the ~8% edge cases.
        pinned = float(common[0]) if common else 0.0
        element_oxi[redox] = pinned
        flags.append(f"out_of_range_pinned:{redox}={redox_oxi:.2f}->{pinned:g}")
        return (element_oxi, {"redox_element": redox, "redox_oxi": pinned,
                              "charge_imbalance": fixed_charge + pinned * doped[redox],
                              "source": "redox_pinned_fallback", "flags": flags})
    element_oxi[redox] = redox_oxi
    return (element_oxi, {"redox_element": redox, "redox_oxi": redox_oxi,
                          "charge_imbalance": fixed_charge, "source": "redox_balance",
                          "flags": flags})


def site_oxidation_states(structure, parent_comp):
    """Per-site occupancy-weighted oxidation for a (possibly disordered) Structure — each
    site is the occupancy-weighted mix of its species. Returns (list[float], info)."""
    doped = dict(structure.composition.get_el_amt_dict())
    element_oxi, info = assign_oxidation(doped, parent_comp)
    site_oxi = [sum(occ * element_oxi.get(getattr(sp, "symbol", str(sp)), 0.0)
                    for sp, occ in site.species.items())
                for site in structure]
    return site_oxi, info


def decorate_structure(structure, parent_comp):
    """Return (decorated_structure, info): a COPY of ``structure`` with per-element
    oxidation states applied via ``add_oxidation_state_by_element`` (fractional ok), so the
    graph builder's explicit-oxidation path reads the doping-aware states. The redox center
    carries the doping charge; mixed sites read between their species' states."""
    element_oxi, info = assign_oxidation(
        dict(structure.composition.get_el_amt_dict()), parent_comp)
    decorated = structure.copy()
    decorated.add_oxidation_state_by_element(element_oxi)
    return decorated, info


def _parse_parent_composition(value) -> dict:
    """Parse a 3DSC `reduced_cell_formula_2` cell — a dict-repr string like
    "{'Sr': 1.0, 'Ge': 2.0, 'Pd': 2.0}" — into {symbol: amount}."""
    import ast
    return {str(k): float(v) for k, v in ast.literal_eval(str(value)).items()}


def parent_composition_map(csv_path: str, cif_col: str = "cif",
                           parent_col: str = "reduced_cell_formula_2") -> dict:
    """{cif basename -> parent composition dict} from the 3DSC master CSV, for keying the
    undoped parent of each doped structure during ingestion. Rows that don't parse are
    skipped (logged count)."""
    import os
    import pandas as pd

    df = pd.read_csv(csv_path)
    out, skipped = {}, 0
    for _, r in df.iterrows():
        try:
            out[os.path.basename(str(r[cif_col]))] = _parse_parent_composition(r[parent_col])
        except Exception:  # noqa: BLE001 — a malformed parent cell just falls back to builder guess
            skipped += 1
    if skipped:
        print(f"  oxidation_doping: {skipped} rows had unparseable {parent_col} (skipped)")
    return out
