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
import functools

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


def _first_positive_common(common) -> float | None:
    """First POSITIVE common oxidation state. pymatgen lists some metalloids' negative
    state first (Sn/Ge/Pb -4), but as a cation here they take a positive state — and an
    accidental negative would create a cation+anion site the builder rejects."""
    positive = [c for c in common if c > 0]
    if positive:
        return float(positive[0])
    return float(common[0]) if common else None


# Oxide-context override where pymatgen's common_oxidation_states lists a LOWER state first
# but the element is virtually always higher in an oxide (Hg2+ not Hg2^2+, Tl3+ not Tl1+,
# GeO2 Ge4+). Only a fallback: the primary path takes oxidations from the parent via
# oxi_state_guesses (data-driven), which already yields these correctly for framework ions.
_OXIDE_PREF = {"Hg": 2.0, "Tl": 3.0, "Ge": 4.0}


def _pinned_cation_oxi(sym: str) -> float:
    """Definite oxidation for a fixed-valence cation (FALLBACK for ions absent from the
    parent oxi_state_guesses). Group 3 + lanthanoids -> +3, alkali -> +1, alkaline-earth
    -> +2, oxide-preferred override, else the element's first positive common state."""
    el = Element(sym)
    if el.is_alkali:
        return 1.0
    if el.is_alkaline:
        return 2.0
    if el.is_lanthanoid or el.group == 3:
        return 3.0
    if sym in _OXIDE_PREF:
        return _OXIDE_PREF[sym]
    v = _first_positive_common(el.common_oxidation_states)
    return v if v is not None else 0.0


def _dopant_oxi(sym: str, replaced_host: str | None, host_oxi: float | None = None) -> float:
    """Oxidation of a dopant element. Single common state -> that. Multiple (e.g. Ce 3/4)
    -> the aliovalent one that DIFFERS from the host it replaces, i.e. the state that
    actually creates doping (Ce->4 replacing Nd->3 = electron doping). `host_oxi` is the
    replaced host's oxidation from the parent reference when available (else pinned)."""
    common = Element(sym).common_oxidation_states
    if not common:
        return _pinned_cation_oxi(sym)
    if len(common) == 1 or replaced_host is None:
        v = _first_positive_common(common)
        return v if v is not None else float(common[0])
    if host_oxi is None:
        host_oxi = _pinned_cation_oxi(replaced_host)
    # A redox TM substituting on another redox-TM site (Co->Fe, Ni->Cu) is typically
    # ISOVALENT — the doping is band-filling, not a formal-oxidation change — so it takes
    # the state nearest the host's, NOT the aliovalent one (which is right for a lanthanide
    # like Ce->Nd creating electron doping).
    if _is_redox_candidate(sym) and replaced_host is not None and _is_redox_candidate(replaced_host):
        return float(min(common, key=lambda c: abs(c - host_oxi)))
    aliovalent = [c for c in common if abs(c - host_oxi) > 1e-9 and c > 0]
    return float(aliovalent[0]) if aliovalent else _first_positive_common(common)


@functools.lru_cache(maxsize=8192)
def _oxide_reference_key(items):
    """(cached) Most-probable integer oxidation per element for an integer cell, via
    pymatgen's ICSD-probability charge-balancing. `items` is a sorted ((sym, int_amt), ...).
    Returns a frozenset of (sym, oxi) or empty when it can't balance."""
    amts = {s: a for s, a in items if a > 0}
    if not amts:
        return frozenset()
    try:
        guesses = Composition(amts).oxi_state_guesses(target_charge=0)
    except Exception:                                # noqa: BLE001 — fractional / no balance
        return frozenset()
    return frozenset((str(k), float(v)) for k, v in guesses[0].items()) if guesses else frozenset()


def _parent_reference_oxi(parent: dict) -> dict:
    """Data-driven reference oxidation per PARENT (undoped) element from oxi_state_guesses.
    This is the general fix: it yields Ru+5/Cu+2 (ruthenocuprate), Hg+2, Tl+3, Fe+2/As-3
    (Fe-based) with NO per-element hardcoding — every fixed ion (incl. a second redox TM)
    gets its physically-correct state, so only the true doping residual lands on the center.
    {} when the parent isn't a clean integer cell (-> per-element fallbacks)."""
    items = tuple(sorted((s, round(v)) for s, v in parent.items() if round(v) > 0))
    return {s: o for s, o in _oxide_reference_key(items)}


def assign_oxidation(doped_comp: dict, parent_comp: dict):
    """Per-element oxidation for a doped composition. Returns (element_oxi, info).

    Fixed ions take their most-probable oxidation from the PARENT (oxi_state_guesses,
    data-driven & general); the doping residual is absorbed by the redox-active transition
    metal, chosen as the first candidate whose charge-balance solve lands in a physical
    range (a 'flip to another TM' guard against a false center). info: redox_element,
    redox_oxi, charge_imbalance, source, flags."""
    doped = {str(k): float(v) for k, v in doped_comp.items()}
    parent = {str(k): float(v) for k, v in parent_comp.items()}
    flags = []

    anions = [e for e in doped if e in ANION_OXI]
    if not anions:                                   # all-metal -> oxidation ill-defined
        return ({e: 0.0 for e in doped},
                {"redox_element": None, "source": "intermetallic", "flags": ["intermetallic"]})

    ref = _parent_reference_oxi(parent)              # {el: oxi} from the parent, or {}
    dopants = [e for e in doped if e not in parent]
    host_drops = {e: parent.get(e, 0.0) - doped.get(e, 0.0)
                  for e in parent if e not in ANION_OXI}
    replaced_host = max(host_drops, key=host_drops.get) if host_drops else None

    def fixed_oxi(e):
        """Oxidation of a non-center ion: anion pinned, parent ion from the ref guess,
        dopant aliovalent, else per-element fallback."""
        if e in ANION_OXI:
            return ANION_OXI[e]
        if e in ref:                                 # data-driven parent reference (general)
            return ref[e]
        if e in dopants:
            return _dopant_oxi(e, replaced_host, ref.get(replaced_host))
        return _pinned_cation_oxi(e)

    host_redox = [e for e in doped if e not in dopants and _is_redox_candidate(e)]
    # Priority: non-coinage before Ag/Au; most ABUNDANT host TM (the framework/active layer —
    # Cu in a ruthenocuprate, majority Fe over minority Co); then higher group; then symbol.
    host_redox.sort(key=lambda s: (s in COINAGE, -doped[s], -Element(s).group, s))
    if len(host_redox) > 1:
        flags.append(f"multi_redox_candidate:{host_redox}")

    def _assign_with_center(center):
        eo, fixed = {}, 0.0
        for e, amt in doped.items():
            if e == center:
                continue
            o = fixed_oxi(e)
            eo[e] = o
            fixed += amt * o
        return eo, fixed

    if not host_redox:                               # ionic but no TM to absorb charge
        eo, fixed = _assign_with_center(None)
        flags.append("no_redox_center")
        return (eo, {"redox_element": None, "charge_imbalance": fixed,
                     "source": "pinned_only", "flags": flags})

    # Try each redox candidate as the doping-absorbing center; accept the first whose solve
    # is physical. With a correct parent reference the majority TM balances first pass; the
    # flip only triggers when the primary center would go unphysical (false-center guard).
    for cand in host_redox:
        eo, fixed = _assign_with_center(cand)
        solve = -fixed / doped[cand]
        # Plausibility from the ICSD-observed range (wider than common_oxidation_states) so
        # cluster compounds like Chevrel Mo (~+2.33) aren't wrongly rejected, while true
        # garbage (a false center, e.g. Ag=+11) still is.
        states = Element(cand).icsd_oxidation_states or Element(cand).common_oxidation_states
        lo, hi = (min(states), max(states)) if states else (0, 6)
        if lo - 1.0 <= solve <= hi + 1.0:
            eo[cand] = solve
            if cand != host_redox[0]:
                flags.append(f"redox_center_flipped:{host_redox[0]}->{cand}")
            return (eo, {"redox_element": cand, "redox_oxi": solve,
                         "charge_imbalance": fixed + solve * doped[cand],
                         "source": "redox_balance", "flags": flags})

    # No candidate balances in range -> pin the primary center, accept a residual (flagged).
    cand = host_redox[0]
    eo, fixed = _assign_with_center(cand)
    common = Element(cand).common_oxidation_states
    pinned = float(common[0]) if common else 0.0
    eo[cand] = pinned
    flags.append(f"out_of_range_pinned:{cand}={-fixed/doped[cand]:.2f}->{pinned:g}")
    return (eo, {"redox_element": cand, "redox_oxi": pinned,
                 "charge_imbalance": fixed + pinned * doped[cand],
                 "source": "redox_pinned_fallback", "flags": flags})


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
