"""SuperCon coverage expansion (user, 2026-09-10): every SuperCon composition the
index does NOT already carry, matched to a real MP parent and synthetically doped
into a usable entry.

Motivation: the Matthias valence-dome test needs the transition-metal solid
solutions (Ti-V 21, Nb-Zr 19, Nb-Ta 36, Cr-V 15 ... 613 non-dilute binaries) that
3DSC dropped; the same machinery recovers the other 9k un-covered rows.

Reuses the NEMAD-expansion pipeline verbatim: formula_match.formula_similarity
(3DSC totreldiff tiers) for parent selection and synth_dope.synth_dope_one
(3DSC synthetic_doping internals) for the doping, so entries are built exactly
like the existing V4/V5 rows.

  candidates   SuperCon - index coverage -> supercon_candidates.csv + need_systems.txt
  pool         MP crystal pool for those chemical systems (lightweight fields)
  match-dope   top-K parent match + synth-dope -> cifs_v7_supercon/ + SC_MP_V7_source.csv
  graphs       id_prop csv -> main.py build-db --kind cgv4 (v45 recipe) -> graphs + index
  report       coverage / tier / doping breakdown, and the Matthias-series census
  python scripts/supercon_expansion/build_supercon_v7.py <cmd> [--limit N] [--workers N]
"""
import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(_sys.argv[0] if __name__ == "__main__" else __file__))
_ROOT = _os.path.dirname(_os.path.dirname(_HERE))
for _p in (_ROOT, _HERE, _os.path.join(_ROOT, "scripts", "nemad_expansion")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse  # noqa: E402
import glob  # noqa: E402
import heapq  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import warnings  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
from pymatgen.core import Composition  # noqa: E402

MP = _os.path.join(_ROOT, "database", "datafiles", "MP")
SCDB = _os.path.join(_ROOT, "database", "datafiles", "SC_EXPAND")
SUPERCON = _os.path.join(MP, "SuperCon_Stanev2018.csv")
INDEX_V45 = _os.path.join(MP, "SC_MP_V4_doped_v45.pickle")
CAND = _os.path.join(SCDB, "supercon_candidates.csv")
SYSTEMS = _os.path.join(SCDB, "need_systems.txt")
POOL = _os.path.join(SCDB, "mp_crystal_pool.pickle")
SOURCE = _os.path.join(MP, "SC_MP_V7_supercon_source.csv")
CIFS = _os.path.join(MP, "cifs_v7_supercon")
IDPROP = _os.path.join(SCDB, "id_prop_v7.csv")
GRAPHS = _os.path.join(MP, "graphs_v4_v7_supercon")
INDEX_OUT = _os.path.join(MP, "SC_MP_V7_supercon")
K = 16


def _subsets(els, min_size=2, max_drop=3):
    """Chemical sub-systems a parent may live in. One element deep is not enough:
    La1.85Sr0.15Cu0.875Ni0.125O4 (5 elements) needs La2CuO4 (3) — two dopants on
    two sublattices, which multi_sub_dope_one handles but the old S / S-{e}
    search never offered. Bounded by max_drop so 7-8 element rows stay cheap."""
    from itertools import combinations
    els = sorted(els)
    lo = max(min_size, len(els) - max_drop)
    out = []
    for k in range(len(els), lo - 1, -1):
        out.extend(combinations(els, k))
    return out


def norm_key(f):
    """Composition -> normalized fractional key (the coverage identity)."""
    try:
        c = Composition(f).fractional_composition.get_el_amt_dict()
    except Exception:  # noqa: BLE001
        return None
    return "|".join(f"{e}{round(v, 3)}" for e, v in sorted(c.items()))


def cmd_candidates(**_):
    _os.makedirs(SCDB, exist_ok=True)
    sc = pd.read_csv(SUPERCON)
    idx = pd.read_pickle(INDEX_V45)
    idx["formula"] = idx.id.str.split("-MP-").str[0].str.split("-ICSD-").str[0]
    have = {k for k in idx.formula.map(norm_key) if k}
    sc["k"] = sc.name.map(norm_key)
    sc = sc[sc.k.notna()].copy()
    # SuperCon lists repeats of the same composition (different reports): keep the
    # MEDIAN Tc and record the spread, mirroring the NEMAD weight/iqr convention.
    g = sc.groupby("k").agg(formula=("name", "first"), tc=("Tc", "median"),
                            n_reports=("Tc", "size"), tc_iqr=("Tc", lambda s: float(s.quantile(.75) - s.quantile(.25))))
    miss = g[~g.index.isin(have)].reset_index()
    # SuperCon writes unspecified oxygen as "...OY" / "OX" / "OZ"; pymatgen parses
    # the trailing Y as YTTRIUM, injecting a spurious cation (47 such entries were
    # built on 2026-09-10 before this filter, e.g. Bi1.8Pb0.2Sr1.2La0.8Cu1OY).
    ph = miss.formula.str.contains(r"O[YXZ]$", regex=True)
    if ph.any():
        print(f"  dropping {int(ph.sum())} rows with SuperCon's unspecified-oxygen placeholder (O_y/O_x/O_z)", flush=True)
        miss = miss[~ph]
    miss["chemsys"] = miss.formula.map(lambda f: "-".join(sorted(Composition(f).get_el_amt_dict())))
    miss["nel"] = miss.chemsys.str.count("-") + 1
    miss["weight"] = 1.0
    miss = miss[["formula", "tc", "n_reports", "tc_iqr", "chemsys", "nel", "k", "weight"]]
    miss.to_csv(CAND, index=False)
    # The matcher accepts a parent whose element set is the target's set OR that set
    # minus ONE element, so the pool must cover those sub-systems too — otherwise a
    # solid solution like Co0.004Ir0.996 finds nothing, because its real parent is
    # elemental Ir (chemsys "Ir"), not "Co-Ir" (the 2026-09-10 empty-parent bug:
    # 1 of 25 TM binaries had any parent at all).
    need = set()
    for cs in miss.chemsys:
        els = sorted(cs.split("-"))
        for sub in _subsets(els):
            need.add("-".join(sub))
        for e in els:
            need.add(e)              # elemental parents: the solid-solution lattices
    with open(SYSTEMS, "w") as f:
        f.write("\n".join(sorted(need)) + "\n")
    print(f"SuperCon unique compositions {len(g)}; index covers {len(have)}; "
          f"candidates (not covered): {len(miss)} across {miss.chemsys.nunique()} chemical systems "
          f"({len(need)} systems to fetch incl. sub-systems)", flush=True)
    print(f"  Tc>0 {int((miss.tc > 0).sum())}, Tc>10 K {int((miss.tc > 10).sum())}, by n_elements "
          f"{miss.nel.value_counts().sort_index().to_dict()}", flush=True)
    print(f"  -> {CAND}\n  -> {SYSTEMS}", flush=True)


def cmd_pool(**_):
    from mp_api.client import MPRester
    key = _os.environ.get("MP_API_KEY") or _sys.exit("error: set MP_API_KEY (env-only)")
    systems = [l.strip() for l in open(SYSTEMS) if l.strip()]
    have = {}
    if _os.path.exists(POOL):                      # resumable
        prev = pd.read_pickle(POOL)
        have = {r.material_id: r for r in prev.itertuples()}
        done = set(prev.chemsys) if "chemsys" in prev.columns else set()
        systems = [s for s in systems if s not in done]
    print(f"pool: {len(systems)} chemical systems to fetch ({len(have)} materials banked)", flush=True)
    rows = [] if not have else pd.read_pickle(POOL).to_dict("records")
    CH = 200
    with MPRester(key) as mpr:
        for i in range(0, len(systems), CH):
            chunk = systems[i:i + CH]
            try:
                docs = mpr.materials.summary.search(
                    chemsys=chunk, fields=["material_id", "formula_pretty", "composition_reduced",
                                           "energy_above_hull", "chemsys"])
            except Exception as e:  # noqa: BLE001
                print(f"  chunk {i}: {str(e)[:90]}", flush=True)
                continue
            for d in docs:
                cr = d.composition_reduced
                cd = {str(el): float(amt) for el, amt in (cr.items() if hasattr(cr, "items") else dict(cr).items())}
                rows.append(dict(material_id=str(d.material_id), formula_pretty=str(d.formula_pretty),
                                 reduced=cd, chemsys=str(d.chemsys),
                                 eah=float(d.energy_above_hull) if d.energy_above_hull is not None else 9.9))
            for s in chunk:
                rows.append(dict(material_id=f"__done__{s}", formula_pretty="", reduced={}, chemsys=s, eah=9.9))
            df = pd.DataFrame(rows).drop_duplicates("material_id")
            df.to_pickle(POOL)                     # checkpoint every chunk
            print(f"  {min(i + CH, len(systems))}/{len(systems)} systems -> {len(df)} rows", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("material_id")
    df.to_pickle(POOL)
    print(f"pool done: {int((~df.material_id.str.startswith('__done__')).sum())} MP materials -> {POOL}", flush=True)


METALS = set("Li Be Na Mg Al K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn "
             "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Th U "
             "Ge Si Sb".split())   # metalloids that form substitutional solid solutions (Al-Ge, Al-Si, Sn-Sb):
                                  # without them an Al0.96Ge0.04 target never saw elemental Al as a host


def alloy_dope_one(structure, target_formula, max_sites=64):
    """Substitutional solid solution on a CHEMICALLY UNIFORM parent lattice: every
    site of a pure-element parent takes the target's fractional composition.

    This is the case 3DSC's synthetic_doping rejects with "different scaling of
    fixed sites" (10.5k of the 11.5k failures on 2026-09-10): an alloy like
    Nb0.5Ti0.5 has no fixed sublattice to preserve — it IS the doped sublattice.
    Restricted to a single-element parent so no ordered compound gets its
    sublattices smeared, and to metals, where substitutional disorder is the
    physical structure (bcc Nb-Ti, hcp Mo-Re, fcc Ir-Pd ...).
    """
    from pymatgen.core import Structure
    st = structure.copy()
    st.remove_oxidation_states()
    parent_els = set()
    for site in st:
        parent_els |= {getattr(e, "symbol", str(e)) for e in site.species}
    if len(parent_els) != 1:
        return None, "alloy: parent not single-element"
    tgt = Composition(target_formula).fractional_composition.get_el_amt_dict()
    tgt = {e: v for e, v in tgt.items() if v > 1e-6}
    if not parent_els <= set(tgt):
        return None, "alloy: parent element not in target"
    tag = "alloy_solid_solution"
    if not set(tgt) <= METALS:
        # a dilute interstitial (O/N/H/C < 3 %) has no site in a metal lattice;
        # build the host alloy and record the dropped impurity in the tag
        non = {e: v for e, v in tgt.items() if e not in METALS}
        if sum(non.values()) >= 0.03:
            return None, "alloy: non-metal in target"
        tgt = {e: v for e, v in tgt.items() if e in METALS}
        z = sum(tgt.values()); tgt = {e: v / z for e, v in tgt.items()}
        tag = "alloy_solid_solution(interstitial_dropped)"
    if len(st) > max_sites:
        return None, "alloy: parent cell too large"
    if len(tgt) == 1:
        return st, "identical" if tag == "alloy_solid_solution" else tag
    new = Structure(st.lattice, [dict(tgt)] * len(st), st.frac_coords)
    got = new.composition.fractional_composition.get_el_amt_dict()
    if max(abs(got.get(e, 0.0) - tgt.get(e, 0.0)) for e in set(got) | set(tgt)) > 1e-4:
        return None, "alloy: composition mismatch"
    return new, tag


ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "H"}


def _role(el):
    return "anion" if el in ANIONS else "cation"


def multi_sub_dope_one(structure, target_formula, tol=0.02, max_sites=200):
    """Multi-sublattice substitutional doping — the case 3DSC rejects with
    "different scaling of fixed sites" (74% of the 2026-09-10 parent-found
    failures, overwhelmingly cuprates): its doper handles ONE doping pair, but
    La1.85Sr0.15Cu0.875Ni0.125O4 needs Sr on the La sublattice AND Ni on the Cu
    sublattice at once, plus an oxygen occupancy that is not 1.

    Method: group the parent's sites into per-element sublattices; assign every
    target element absent from the parent to the most similar parent sublattice
    (same ion role, nearest electronegativity + relative radius); fix the cell
    scale by filling the CATION sublattices exactly; put the remaining anion
    content on the anion sites as a partial occupancy (deficiency allowed,
    interstitials are not — those need the ICSD parent route). Reject unless the
    resulting composition reproduces the target to `tol` in fractional terms.
    """
    from pymatgen.core import Element, Structure
    st = structure.copy()
    st.remove_oxidation_states()
    if len(st) > max_sites:
        return None, "multisub: parent cell too large"
    sub = defaultdict(list)                      # parent element -> site indices
    for i, site in enumerate(st):
        occ = {getattr(e, "symbol", str(e)): float(v) for e, v in site.species.items()}
        if not occ:
            return None, "multisub: empty site"
        # A partially occupied parent site is a CAPACITY, not a composition: an
        # ICSD 123 refinement carries the O(5) chain site at occupancy ~0.01-0.2,
        # and that site is exactly the extra anion room the excess-O targets need.
        # Label the sublattice by the site's majority species and let the LP refill it.
        sub[max(occ, key=occ.get)].append(i)
    tgt = {e: v for e, v in Composition(target_formula).fractional_composition.get_el_amt_dict().items() if v > 1e-6}
    if not set(sub) & set(tgt):
        return None, "multisub: no shared element"

    def prop(e):
        el = Element(e)
        return (el.X or 1.5), (el.atomic_radius or 1.4)

    cat_sites = sum(len(v) for k, v in sub.items() if _role(k) == "cation")
    cat_frac = sum(v for e, v in tgt.items() if _role(e) == "cation")
    if cat_sites == 0 or cat_frac <= 0:
        return None, "multisub: no cation sublattice"
    # Cation-deficient formulas (YBa2Cu2.94O6.94) cannot fill every cation site
    # with their own atoms and the only fillers left are implausible; allow up to
    # 3 % cation vacancies by trying progressively smaller fillings.
    for _fill in (1.0, 0.99, 0.98, 0.97):
        res_ = _distribute(sub, tgt, cat_sites / cat_frac * _fill)
        if res_ is not None:
            break
    else:
        return None, "multisub: no feasible site distribution"
    occ, banned = res_
    species = [None] * len(st)
    for p, d in occ.items():
        for i in sub[p]:
            species[i] = dict(d)
    new = Structure(st.lattice, species, st.frac_coords)
    got = new.composition.fractional_composition.get_el_amt_dict()
    if max(abs(got.get(e, 0.0) - tgt.get(e, 0.0)) for e in set(got) | set(tgt)) > tol:
        return None, "multisub: composition mismatch"
    return new, "multi_sublattice"


def _distribute(sub, tgt, total):
    """Transportation LP for one cell filling; None when infeasible."""
    from pymatgen.core import Element
    from scipy.optimize import linprog

    def prop(e):
        el = Element(e)
        return (el.X or 1.5), (el.atomic_radius or 1.4)
    # Distribute each target element over the COMPATIBLE sublattices instead of
    # committing it to the single nearest one: a dopant often splits (La sits on
    # both the Y and the Ba site of a 123), which a greedy assignment reports as
    # "sublattice over-filled" (900+ of the 2026-09-10 failures). Transportation
    # LP: supply = target atoms, capacity = sites (exact for cations, <= for
    # anions), cost = chemical dissimilarity.
    from scipy.optimize import linprog
    els, subs = list(tgt), list(sub)
    idx = {(e, p): k for k, (e, p) in enumerate([(e, p) for e in els for p in subs])}
    cost, big, banned = [], 1e3, set()
    # A cation may only sit on a chemically PLAUSIBLE sublattice: pricing alone
    # let La onto a Cu site when the target was infeasible on the real sites.
    # dissimilarity = |dX| + |dr|/r; 0.9 separates Ca/Sr/La-on-A-site (0.1-0.5)
    # from La/Y-on-Cu (1.0-1.2).
    MAXD = 0.9
    for e in els:
        xe, re_ = prop(e)
        for p in subs:
            d = abs(prop(p)[0] - xe) + abs(prop(p)[1] - re_) / prop(p)[1]
            if _role(p) != _role(e) or (e != p and d > MAXD):
                banned.add((e, p)); cost.append(big)
            else:
                cost.append(d + (0.0 if e == p else 0.25))
    A_eq, b_eq, A_ub, b_ub = [], [], [], []
    for e in els:                                 # every target atom is placed
        row = [0.0] * len(cost)
        for p in subs:
            row[idx[(e, p)]] = 1.0
        A_eq.append(row); b_eq.append(tgt[e] * total)
    for p in subs:                                # site capacity
        row = [0.0] * len(cost)
        for e in els:
            row[idx[(e, p)]] = 1.0
        if _role(p) == "cation":
            A_ub.append(row); b_ub.append(float(len(sub[p])))
            A_ub.append([-x for x in row]); b_ub.append(-float(len(sub[p])) * 0.97)
        else:
            A_ub.append(row); b_ub.append(float(len(sub[p])) * (1.0 + ANION_TOL))
    res = linprog(cost, A_ub=A_ub or None, b_ub=b_ub or None, A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, None), method="highs")
    if not res.success:
        return None
    occ = defaultdict(dict)
    for e in els:
        for p in subs:
            v = res.x[idx[(e, p)]]
            if v > 1e-6:
                if (e, p) in banned:
                    return None
                occ[p][e] = occ[p].get(e, 0.0) + v / len(sub[p])
    for p in subs:
        if p not in occ:
            return None
        if sum(occ[p].values()) > 1.0 + 1e-3:
            if _role(p) == "anion" and sum(occ[p].values()) <= 1.0 + ANION_TOL + 1e-3:
                z = sum(occ[p].values()); occ[p] = {e: v / z for e, v in occ[p].items()}
            else:
                return None
    return occ, banned


def _redox_centre(els):
    """The element a family's superconductivity lives on. A parent that lacks it
    cannot host the target's physics however well the site roles fit (audit
    2026-09-14: Th1Ca3Ba1Cu3O9 was built on BaThO3, V-doped Bi-2223 on
    Sr3Ca6V7BiO28 - 489 such rows)."""
    e = set(els)
    if "Cu" in e and "O" in e:
        return "Cu"
    if "Fe" in e and e & {"As", "P", "Se", "Te", "S"}:
        return "Fe"
    if "Ni" in e and "O" in e:
        return "Ni"
    return None


def _anion_ratio(cd):
    a = sum(v for e, v in cd.items() if e in ANIONS)
    k = sum(v for e, v in cd.items() if e not in ANIONS)
    return a / k if k else float("nan")


RARE_EARTHS = {"Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}


def _canon_re(els):
    """Element set with every rare earth collapsed to one token: RE-substituted
    frameworks (Ho-123, Eu-124, Nd-214 ...) share a template and differ only by
    an isovalent cation. 941 of the 1,593 candidates left unbuilt on 2026-09-14
    matched a template this way and nothing else."""
    return frozenset("RE" if e in RARE_EARTHS else e for e in els)


def _cation_fit(cd_t, cd_p):
    """How well the target's cation proportions can be laid onto the parent's
    cation sublattices, ignoring chemistry (the doper checks that): the best L1
    distance between the parent's sorted sublattice fractions and the target's
    element fractions merged into that many groups. 0 = perfect. Ranking by
    anion ratio + shared elements alone put Sr4Ca10Cu24O41 on Sr19Cu19O41 and
    Bi-2201 derivatives on La8Cu7O19 (2026-09-14 diagnosis: 44/60 failures were
    'element has no room', i.e. the framework was wrong, not the chemistry)."""
    from itertools import combinations
    t = sorted((v for e, v in cd_t.items() if e not in ANIONS), reverse=True)
    p = sorted((v for e, v in cd_p.items() if e not in ANIONS), reverse=True)
    if not t or not p:
        return 9.9
    t = [x / sum(t) for x in t]; p = [x / sum(p) for x in p]
    n, k = len(t), len(p)
    if n < k:                       # fewer target elements than sublattices: some element must span
        return 0.1                  # two sites (legitimate, e.g. Y on both the Y and the Ca site) - mild penalty
    if n == k:
        return sum(abs(a - b) for a, b in zip(t, p))
    # more target elements than sublattices: (n-k) elements must sit as foreign
    # occupants on someone else's site - a single-sublattice parent (Fe3O) would
    # otherwise "fit" Dy1Fe1As1O0.85 perfectly while the real template LaFeAsO
    # sat one rare-earth swap away
    merge_penalty = 0.2 * (n - k)
    best = 9.9
    # partition the n target fractions into k contiguous-or-not groups: brute force
    # over assignments of the (n-k) smallest fractions onto the k groups
    heads, tails = t[:k], t[k:]
    from itertools import product
    for assign in product(range(k), repeat=len(tails)):
        g = list(heads)
        for x, j in zip(tails, assign):
            g[j] += x
        best = min(best, sum(abs(a - b) for a, b in zip(sorted(g, reverse=True), p)))
        if best < 1e-9:
            break
    return best + merge_penalty


RATIO_CAP = 1.0        # parents whose anion/cation ratio differs more than this are a different compound class
ANION_TOL = 0.02       # anion sublattice may be over-filled by this fraction (nominal O7 on an O7 host with a
                       # slight cation deficiency lands at 1.005-1.02); occupancies are then renormalized to 1 and
                       # the result must still pass the composition-fidelity tolerance
BANNED = _os.path.join(SCDB, "banned_keys.txt")   # composition keys never to build (audited typos)


def cmd_match_dope(limit=None, **_):
    from formula_match import chem_dict, formula_similarity
    from synth_dope import synth_dope_one
    from pymatgen.io.cif import CifWriter
    from mp_api.client import MPRester
    _os.makedirs(CIFS, exist_ok=True)
    pool = pd.read_pickle(POOL)
    pool = pool[~pool.material_id.str.startswith("__done__")].copy()
    pool["cd"] = pool["reduced"].map(chem_dict)
    by_sys = defaultdict(list)
    for r in pool.itertuples():
        by_sys[frozenset(r.cd)].append((r.material_id, r.cd, r.eah, r.formula_pretty))
    # ---- hand-picked ICSD parents (docs/data_curation/icsd_parents_wanted.md) ----
    # Keyed by the SUBLATTICE elements (majority species per site), not the whole
    # refinement: a Na-substituted 123 is a YBa2Cu3O7 host whose Na is an impurity
    # on the Y site, and its partially occupied O(5) site is exactly the extra
    # anion room the excess-O targets need. Tier 0 -> tried before any MP parent.
    from pymatgen.core import Structure as _Struct
    icsd = {}
    for cif in sorted(glob.glob(_os.path.join(MP, "ICSD_Parent_Cifs", "*.cif"))):
        code = _os.path.basename(cif).replace("EntryWithCollCode", "").replace(".cif", "")
        try:
            st_i = _Struct.from_file(cif)
            st_i.remove_oxidation_states()
        except Exception as e:  # noqa: BLE001
            print(f"  icsd {code}: unreadable ({str(e)[:50]})", flush=True)
            continue
        labels = defaultdict(float)               # majority label -> number of sites
        for site in st_i:
            occ = {getattr(e, "symbol", str(e)): float(v) for e, v in site.species.items()}
            if occ:
                labels[max(occ, key=occ.get)] += 1.0
        mid = f"icsd-{code}"
        icsd[mid] = st_i
        # site COUNTS per sublattice, not 1.0 each: _cation_fit compares these
        # fractions with the target's, and a refinement's formula (Sr2Ca0.5Y0.5...)
        # has more elements than its structure has sublattices
        by_sys[frozenset(labels)].append((mid, dict(labels), 0.0, st_i.composition.reduced_formula))
    print(f"  ICSD parents loaded: {len(icsd)}", flush=True)
    by_re = defaultdict(list)                     # RE-canonical element set -> parents
    for sysk, lst in list(by_sys.items()):
        if sysk & RARE_EARTHS:
            for t in lst:
                by_re[_canon_re(sysk)].append(t)
    cand = pd.read_csv(CAND)
    done = set()
    if _os.path.exists(SOURCE):                    # resumable
        done = set(pd.read_csv(SOURCE)["k"])
        cand = cand[~cand.k.isin(done)]
    if _os.path.exists(BANNED):
        banned = {l.strip() for l in open(BANNED) if l.strip()}
        nb = int(cand.k.isin(banned).sum())
        cand = cand[~cand.k.isin(banned)]
        print(f"  banned (audited typos) skipped: {nb}", flush=True)
    if limit:
        cand = cand.head(limit)
    print(f"match-dope: {len(cand)} candidates ({len(done)} already built)", flush=True)
    topk = {}
    swaps = {}                                    # (candidate key, parent id) -> {parent RE: target RE}
    for c in cand.itertuples():
        cd_sc = chem_dict(c.formula)
        S = frozenset(cd_sc)
        subsets = [S] if len(S) == 1 else [frozenset(x) for x in _subsets(sorted(S))]
        centre = _redox_centre(S)
        r_t = _anion_ratio(cd_sc)
        tot = sum(cd_sc.values())
        metal_part = {e: v for e, v in cd_sc.items() if e in METALS}
        all_metal = set(S) <= METALS
        # dilute interstitial in a metal (Nb99.9O0.1, Al0.01Nb0.97O0.02): the host
        # is the ELEMENT, not an oxide of it (audit: these landed on Nb12O29)
        dilute_metal = (not all_metal) and sum(metal_part.values()) / tot >= 0.97

        def gate(cd2):
            """Parent acceptance: must carry the family's redox centre and be of
            the same compound class (anion/cation ratio within RATIO_CAP)."""
            if centre and centre not in cd2:
                return False
            if not all_metal and not dilute_metal:
                r_p = _anion_ratio(cd2)
                if not np.isnan(r_p) and abs(r_p - r_t) > RATIO_CAP:
                    return False
            return True

        acc = []
        for sub in subsets:
            for mid, cd2, eah, pf in by_sys.get(sub, ()):
                if not gate(cd2):
                    continue
                tier, trd = formula_similarity(cd_sc, cd2)
                if np.isnan(tier):
                    continue
                acc.append((int(tier), round(eah, 4), round(trd, 5), mid, pf))
        # all-metal composition (or a metal with a dilute interstitial) -> offer
        # the ELEMENTAL lattices as solid-solution parents, majority first
        if all_metal or dilute_metal:
            major = max(metal_part, key=metal_part.get)
            for e in metal_part:
                for mid, cd2, eah, pf in by_sys.get(frozenset({e}), ()):
                    acc.append((4 if e == major else 5, round(eah, 4), 1.0, mid, pf))
        # Last resort: any GATED parent whose element set is a subset of the
        # target's, ranked by anion/cation-ratio closeness FIRST and shared
        # elements second. Ranking by shared elements alone chose a molecular
        # chloride (Te3MoCl16) over the Chevrel host (Mo6Te8) for Mo6Te6Cl2.
        for sub in subsets:
            for mid, cd2, eah, pf in by_sys.get(sub, ()):
                if str(mid).startswith("icsd-"):
                    if gate(cd2):
                        acc.append((0, 0.0, 1.0, mid, pf))
                    continue
                if not gate(cd2):
                    continue
                r_p = _anion_ratio(cd2)
                dr = 0.0 if (all_metal or np.isnan(r_p)) else abs(r_p - r_t)
                fit = _cation_fit(cd_sc, cd2)
                acc.append((6 + int(round(dr * 4)), round(fit, 2), 0, len(S) - len(sub), round(eah, 4), mid, pf))
        # RE-swapped templates: a gated parent whose elements match the target's
        # once rare earths are interchangeable, and which carries an RE the target
        # lacks. Ranked just below the exact-element fallback. The swap itself is
        # applied to the structure at doping time (see swaps[]).
        if S & RARE_EARTHS:
            tgt_re = [e for e in sorted(S, key=lambda e: -cd_sc[e]) if e in RARE_EARTHS]
            for sub in subsets:
                for mid, cd2, eah, pf in by_re.get(_canon_re(sub), ()):
                    p_re = set(cd2) & RARE_EARTHS
                    if not p_re - S or not gate(cd2):
                        continue
                    if not (set(cd2) - RARE_EARTHS) <= S:
                        continue
                    r_p = _anion_ratio(cd2)
                    dr = 0.0 if np.isnan(r_p) else abs(r_p - r_t)
                    swap = {r: tgt_re[0] for r in p_re - S}
                    swaps[(c.k, mid)] = swap
                    fit = _cation_fit(cd_sc, cd2)
                    # same tier band as the exact fallbacks: a swapped template with a
                    # good cation fit must beat an exact-element parent with a bad one
                    # (Dy-1111 was built on Fe3O while LaFeAsO sat one swap away)
                    acc.append((6 + int(round(dr * 4)), round(fit, 2), 1, len(S) - len(sub), round(eah, 4), mid, pf))
        if acc:
            # separate quotas: the exact-element fallbacks (tier 6-9) would otherwise
            # fill every slot ahead of the RE-swapped templates (tier 10+) and the
            # swaps would never be tried (first pass built 10 of 941)
            exact = [t for t in acc if not (len(t) == 7 and t[2] == 1)]
            swapped = [t for t in acc if len(t) == 7 and t[2] == 1]
            topk[c.k] = sorted(heapq.nsmallest(K, exact) + heapq.nsmallest(K, swapped))
    cand = cand[cand.k.isin(topk)].copy()
    print(f"  candidates with >=1 accepted parent: {len(cand)}", flush=True)
    mids = sorted({t[-2] for lst in topk.values() for t in lst if not str(t[-2]).startswith("icsd-")})
    key = _os.environ.get("MP_API_KEY") or _sys.exit("error: set MP_API_KEY (env-only)")
    print(f"  fetching {len(mids)} parent structures", flush=True)
    cache = dict(icsd)                            # ICSD parents need no fetch
    with MPRester(key) as mpr:
        for i in range(0, len(mids), 400):
            for d in mpr.materials.summary.search(material_ids=mids[i:i + 400], fields=["material_id", "structure"]):
                cache[str(d.material_id)] = d.structure
            print(f"    {min(i + 400, len(mids))}/{len(mids)}", flush=True)
    rows, nfail, reasons = [], 0, defaultdict(int)
    info = {r.k: r for r in cand.itertuples()}
    for n, (k, lst) in enumerate(topk.items()):
        c = info.get(k)
        if c is None:
            continue
        built = False
        for t in lst:
            tier, mid, pf = t[0], t[-2], t[-1]
            eah, trd = (t[1], t[2]) if len(t) == 5 else (t[-3], t[1])
            st0 = cache.get(mid)
            if st0 is None:
                continue
            sw = swaps.get((k, mid))
            if sw:                                # RE-swapped template
                st0 = st0.copy(); st0.remove_oxidation_states()
                st0.replace_species({a: b for a, b in sw.items()})
            try:
                st, reason = synth_dope_one(st0, c.formula)
            except Exception as e:  # noqa: BLE001
                st, reason = None, f"exception:{type(e).__name__}"
            if st is None:
                try:
                    st, reason2 = alloy_dope_one(st0, c.formula)
                except Exception as e:  # noqa: BLE001
                    st, reason2 = None, f"alloy exception:{type(e).__name__}"
                if st is None:
                    try:
                        st, reason3 = multi_sub_dope_one(st0, c.formula)
                    except Exception as e:  # noqa: BLE001
                        st, reason3 = None, f"multisub exception:{type(e).__name__}"
                    if st is None:
                        reasons[reason if "alloy:" not in reason2 else reason3] += 1
                        continue
                    reason = reason3
                else:
                    reason = reason2
            ident = f"{c.formula}-MP-{mid}"
            cifp = _os.path.join(CIFS, f"{ident}.cif")
            try:
                CifWriter(st, symprec=None).write_file(cifp)
                from pymatgen.core import Structure as _S
                _S.from_file(cifp)               # must round-trip: ~5% of written
            except Exception:                     # CIFs were unreadable (occupancy
                reasons["cif unreadable"] += 1    # collisions) and died later in
                if _os.path.exists(cifp):         # the graph build instead
                    _os.remove(cifp)
                continue
            rows.append(dict(id=ident, cif=_os.path.relpath(cifp, _ROOT), tc=c.tc, k=c.k,
                             n_reports=c.n_reports, tc_iqr=c.tc_iqr, parent_formula=pf,
                             tier=tier, totreldiff=trd,
                             doping=reason + (f"(RE-swap {','.join(f'{a}>{b}' for a, b in sw.items())})" if sw else ""),
                             eah=eah))
            built = True
            break
        if not built:
            nfail += 1
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(topk)} built {len(rows)} failed {nfail}", flush=True)
            pd.DataFrame(rows).to_csv(SOURCE + ".part", index=False)
    out = pd.DataFrame(rows)
    if done and _os.path.exists(SOURCE):
        out = pd.concat([pd.read_csv(SOURCE), out], ignore_index=True).drop_duplicates("id")
    out.to_csv(SOURCE, index=False)
    if _os.path.exists(SOURCE + ".part"):
        _os.remove(SOURCE + ".part")
    print(f"match-dope DONE: {len(out)} entries built (this pass {len(rows)}, failed {nfail})", flush=True)
    print(f"  doping: {out.doping.value_counts().to_dict()} | tiers: {out.tier.value_counts().to_dict()}", flush=True)
    print(f"  top failure reasons: {dict(sorted(reasons.items(), key=lambda x: -x[1])[:8])}", flush=True)


def cmd_graphs(**_):
    src = pd.read_csv(SOURCE)
    # build-db --kind cgv4 (no --has-header) consumes "<cif file>,<tc>" rows, cif
    # paths relative to the cif dir — the exact id_prop.csv convention of the v45 wave.
    with open(IDPROP, "w") as f:
        for r in src.itertuples():
            f.write(f"{_os.path.basename(r.cif)},{r.tc}\n")
    print(f"graphs: {len(src)} rows -> {IDPROP}; building cgv4 graphs (v45 recipe)", flush=True)
    cmd = [_sys.executable, "main.py", "build-db", "--kind", "cgv4",
           "--source", IDPROP, CIFS + "/",
           "--output", INDEX_OUT, "--graph-dir", GRAPHS,
           "--oxidation-parent-csv", _os.path.join(MP, "3DSC_MP.csv")]
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=_ROOT, check=True)
    idx = pd.read_pickle(INDEX_OUT + ".pickle")
    print(f"graphs DONE: {len(idx)} rows in {INDEX_OUT}.pickle", flush=True)


def cmd_report(**_):
    TM = {"Sc", "Y", "La", "Lu", "Ti", "Zr", "Hf", "V", "Nb", "Ta", "Cr", "Mo", "W", "Mn", "Tc",
          "Re", "Fe", "Ru", "Os", "Co", "Rh", "Ir", "Ni", "Pd", "Pt"}
    VAL = {"Sc": 3, "Y": 3, "La": 3, "Lu": 3, "Ti": 4, "Zr": 4, "Hf": 4, "V": 5, "Nb": 5, "Ta": 5,
           "Cr": 6, "Mo": 6, "W": 6, "Mn": 7, "Tc": 7, "Re": 7, "Fe": 8, "Ru": 8, "Os": 8,
           "Co": 9, "Rh": 9, "Ir": 9, "Ni": 10, "Pd": 10, "Pt": 10}
    cand = pd.read_csv(CAND)
    if not _os.path.exists(SOURCE):
        print("no source csv yet"); return
    src = pd.read_csv(SOURCE)
    print(f"candidates {len(cand)} -> built {len(src)} ({100 * len(src) / max(len(cand), 1):.0f}%)")
    print(f"  doping {src.doping.value_counts().to_dict()}  tier {src.tier.value_counts().to_dict()}")
    src["formula"] = src.id.str.split("-MP-").str[0]
    def ea(f):
        try:
            c = Composition(f).get_el_amt_dict()
        except Exception:  # noqa: BLE001
            return None
        return sum(VAL[e] * a for e, a in c.items()) / sum(c.values()) if set(c) <= set(VAL) else None
    src["ea"] = src.formula.map(ea)
    tm = src[src.ea.notna()].copy()
    tm["els"] = tm.formula.map(lambda f: "-".join(sorted(Composition(f).get_el_amt_dict())))
    print(f"\ntransition-metal alloy entries recovered: {len(tm)} across {tm.els.nunique()} systems")
    g = tm.groupby("els").agg(n=("id", "size"), ea_min=("ea", "min"), ea_max=("ea", "max"), tc_max=("tc", "max"))
    g["span"] = (g.ea_max - g.ea_min).round(2)
    print(g[g.n >= 4].sort_values("n", ascending=False).round(2).head(25).to_string())
    b = pd.cut(tm.ea, np.arange(3.5, 8.6, 0.25))
    print("\nrecovered Matthias curve (max Tc per e/a bin):")
    print(tm.groupby(b, observed=True).tc.agg(["size", "max", "median"]).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["candidates", "pool", "match-dope", "graphs", "report"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"candidates": cmd_candidates, "pool": cmd_pool, "match-dope": cmd_match_dope,
     "graphs": cmd_graphs, "report": cmd_report}[a.cmd](limit=a.limit, workers=a.workers)
