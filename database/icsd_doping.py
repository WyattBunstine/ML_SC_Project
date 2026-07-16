"""Synthetic doping from oxygen-rich ICSD parent structures for the NEMAD expansion.

Handles the cases 3DSC's `_4_synthetic_doping` (and the MP ground-state parents)
cannot: oxygen-interstitial cuprates where O sits on both ordered plane sites and
partially-occupied chain/apical sites, plus cation co-doping. Two-step per target:

  1. cation substitution  — each dopant element is placed on the site of the host
     it replaces (matched by size class + which host loses that amount), as a
     fractional occupancy; and
  2. oxygen adjustment     — the structural (fully-occupied) plane O is held fixed
     while the partially-occupied chain/apical O sites are varied to hit the target
     total O.

For the REBa2Cu3O(6+x) family every member is isostructural, so a single
orthorhombic template (with a partial chain-O site) is RE-substituted per target.
"""
import warnings
from pymatgen.core import Structure, Composition, Element

ANIONS = {"O", "F", "Cl", "N", "C", "S", "Se", "Te", "Br", "I", "H", "P"}
# large A-site cations vs small B-site cations, for matching a dopant to its host site
LARGE = {"Ba", "Sr", "Ca", "K", "Na", "Rb", "Cs", "Pb", "Bi", "Tl", "Hg",
         "La", "Y", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
         "Er", "Tm", "Yb", "Lu", "Sc"}
SMALL = {"Cu", "Mn", "Fe", "Co", "Ni", "Zn", "Al", "Ga", "Sn", "Ti", "V",
         "Cr", "Mo", "Ru", "Ag", "Mg", "Nb", "W", "Re", "Ir", "Pt", "Pd", "Au", "Ge", "Zr"}


def read_icsd_cif(path):
    """Read an ICSD CIF, tolerating a stray leading non-CIF line."""
    txt = open(path).read()
    if not txt.lstrip().startswith(("#", "data_", "_")):
        txt = "\n".join(txt.splitlines()[1:])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Structure.from_str(txt, fmt="cif")


def re_substitute(structure, new_re, re_set=LARGE):
    """Swap the (single) rare-earth on the RE site of a 123 template for `new_re`.
    Strips oxidation states first so Element keys match (CIFs carry e.g. Nd3+)."""
    st = structure.copy(); st.remove_oxidation_states()
    d = st.composition.get_el_amt_dict()
    # the RE in a REBa2Cu3O7 template = the large cation that is neither Ba nor Cu/O
    cur_re = [e for e in d if e in re_set and e not in ("Ba", "Sr", "Ca")]
    cur_re = [e for e in cur_re if e != new_re]
    if len(cur_re) == 1 and cur_re[0] != new_re:
        st.replace_species({Element(cur_re[0]): Element(new_re)})
    return st


def _pick_host(dopant, cur, des):
    """Which existing element does `dopant` substitute for?"""
    deficit = {e: cur.get(e, 0) - des.get(e, 0) for e in cur}
    if dopant in ANIONS:
        cands = [e for e in cur if e in ANIONS and deficit[e] > 0.01]
        return max(cands, key=lambda e: deficit[e]) if cands else ("O" if "O" in cur else None)
    cls = LARGE if dopant in LARGE else (SMALL if dopant in SMALL else None)
    need = des[dopant] - cur.get(dopant, 0)
    pool = [e for e in cur if e not in ANIONS and deficit[e] > 0.01]
    same = [e for e in pool if cls and e in cls]
    cands = same or pool
    return min(cands, key=lambda e: abs(deficit[e] - need)) if cands else None


def _substitute_on_site(st, host, dopant, amount):
    """Place `amount` of `dopant` onto `host`'s sites, reducing host occupancy so the
    per-site total is preserved (fractional solid-solution)."""
    host_sites = [(i, sum(o for el, o in st[i].species.items() if el.symbol == host))
                  for i in range(len(st))]
    host_sites = [(i, o) for i, o in host_sites if o > 1e-6]
    tot = sum(o for _, o in host_sites)
    if tot < amount - 1e-6:
        return False
    for i, o in host_sites:
        take = amount * (o / tot)
        new = {el: occ for el, occ in st[i].species.items()}
        # reduce host, add dopant
        hk = next(el for el in new if el.symbol == host)
        new[hk] = new[hk] - take
        dk = Element(dopant)
        new[dk] = new.get(dk, 0.0) + take
        new = {el: round(occ, 5) for el, occ in new.items() if occ > 1e-5}
        st[i] = new
    return True


def _adjust_oxygen(st, target_O):
    """Hold fully-occupied O fixed; vary the partially-occupied chain/apical O to reach
    target_O. If the parent is fully oxygen-ordered (no partial site) the least-bonded
    O site is treated as the variable (chain/reservoir) site."""
    o = [i for i in range(len(st)) if any(el.symbol == "O" for el in st[i].species)]
    occ = lambda i: sum(x for el, x in st[i].species.items() if el.symbol == "O")
    var = [i for i in o if occ(i) < 0.999]
    if not var:
        # fully ordered: pick the O site with the fewest cation neighbours (reservoir/chain O)
        try:
            cn = {i: len(st.get_neighbors(st[i], 3.0)) for i in o}
            var = [min(cn, key=cn.get)]
        except Exception:
            var = [o[-1]] if o else []
    fixed = sum(occ(i) for i in o if i not in var)
    chain = target_O - fixed
    # A variable site's O capacity is what its CO-OCCUPANTS leave free (an F-doped
    # site holding F 0.05 can take O up to 0.95) — and adjusting O must PRESERVE
    # those co-occupants: the old wholesale `st[i] = {O: give}` silently erased a
    # site-sharing anion dopant (this killed every Nd2CuO4-xFx build).
    other = lambda i: {el: x for el, x in st[i].species.items() if el.symbol != "O"}
    cap = sum(1.0 - sum(other(i).values()) for i in var)
    if chain < -1e-6 or chain > cap + 1e-6:
        return None, f"O out of range (need chain {chain:.2f}, cap {cap:.1f})"
    rem, drop = chain, []
    for i in var:
        room = 1.0 - sum(other(i).values())
        give = min(room, rem); rem -= give
        new = dict(other(i))
        if give >= 1e-4:
            new[Element("O")] = round(give, 5)
        if new:
            st[i] = new
        else:
            drop.append(i)
    if drop:
        st.remove_sites(drop)
    return st, "ok"


def combined_dope(structure, target_formula, tol=0.04):
    """Dope `structure` (an oxygen-rich parent) to `target_formula`. Returns
    (Structure, reason) on success or (None, reason) on failure."""
    st = structure.copy(); st.remove_oxidation_states()
    tg = Composition(target_formula).get_el_amt_dict()
    cur = st.composition.get_el_amt_dict()
    shared = [e for e in tg if e != "O" and e in cur]
    if not shared:
        return None, "no shared cation"
    # scale by TOTAL cation count (not one reference cation — that would zero out the
    # reference's deficit and hide it as a doping host, e.g. Cu for TM dopants).
    # Scale by the TRUE cation count: every anion is excluded, not just O — an
    # anion dopant (F in Nd2CuO4-xFx) counted as a "cation" skews the scale and
    # every des[] amount with it (Nd 2.0 came out as des 1.88).
    cat_cur = sum(v for e, v in cur.items() if e not in ANIONS)
    cat_tg = sum(v for e, v in tg.items() if e not in ANIONS)
    scale = cat_cur / cat_tg if cat_tg else 1.0
    des = {e: v * scale for e, v in tg.items()}
    # 1) cation / anion substitution for every element that must be ADDED
    dopants = {e: des[e] for e in des
               if e != "O" and (e not in cur or des[e] > cur.get(e, 0) + tol)}
    for D in sorted(dopants, key=lambda e: -(des[e] - cur.get(e, 0))):
        need = des[D] - cur.get(D, 0)
        host = _pick_host(D, cur, des)
        if host is None:
            return None, f"no host for {D}"
        if not _substitute_on_site(st, host, D, need):
            return None, f"cannot place {D} on {host}"
        cur = st.composition.get_el_amt_dict()
    # 2) oxygen — only when substitution hasn't already landed the O target: an
    # anion dopant (F on O) reduces O occupancy in the same move, so re-adjusting
    # would redistribute O across the freshly doped sites for no reason.
    if "O" in des:
        cur_O = st.composition.get_el_amt_dict().get("O", 0.0)
        if abs(cur_O - des["O"]) > tol:
            st, why = _adjust_oxygen(st, des["O"])
            if st is None:
                return None, why
    # 3) validate composition matches target (up to the reference scale)
    got = st.composition.get_el_amt_dict()
    for e in set(des) | set(got):
        if abs(got.get(e, 0) - des.get(e, 0)) > 0.08:
            return None, f"mismatch {e}: {got.get(e,0):.2f} vs {des.get(e,0):.2f}"
    return st, "doped"
