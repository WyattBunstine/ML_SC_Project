"""Structure-type holdouts: hold out a STRUCTURE (cation framework), not an
element set (user, 2026-09-09: "nothing matching that structure is in the
training data, not just the stoichiometry" — but 124 is a different structure
from 123 and stays in training).

Framework = the structure with every anion (O/F/Cl/Br/I/S/Se/Te/N/H) removed
and every cation replaced by one dummy species, reduced to its primitive cell.
Rows sharing an MP parent share a framework by construction (synthetic doping
only substitutes), so one CIF per parent is parsed. Frameworks are compared
with StructureMatcher(FrameworkComparator) at tight tolerances (ltol 0.1,
stol 0.15, 3 deg) — an inert-cation swap (Y->Ho, Bi->Tl) matches, a
different layer stacking (123 vs 124) does not.

Per family:
  holdout  = rows of the existing element-set holdout whose framework is one
             of the family's PROTECTED frameworks (yba: the 123 framework of
             mp-20674 only; others: every framework among the holdout parents)
  exclude  = every other index row whose framework matches a protected one
             (any element set) + curated label suspects for that family
Writes database/datafiles/MP/holdout_struct_<fam>_ids.csv and
exclude_struct_<fam>_ids.csv, plus a match report.
Usage: python scripts/structure_holdout.py [fam ...]
"""
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher, FrameworkComparator

MP = "database/datafiles/MP"
INDEX = f"{MP}/SC_MP_V4_doped_v45.pickle"
ANIONS = {"O", "F", "Cl", "Br", "I", "S", "Se", "Te", "N", "H"}
PROTECT_PARENT = {"yba": ["mp-20674"]}          # 123 only; 124 stays in training
SUSPECTS = {"yba": "docs/data_curation/yba_holdout_label_suspects.csv"}
# The cation framework cannot see the oxygen sublattice, which is what makes
# YBCO-123 (Cu-O chains) a different structure from Tl/Hg/Ru-1212 (MO layers),
# and T' (Nd2CuO4, square-planar Cu) different from T (La2CuO4, octahedral).
# Per-family chemistry filters restore that distinction on top of the match:
CHEM = {
    "yba":    lambda e: not (e & {"Tl", "Hg", "Ru", "Bi", "Pb"}),     # Cu-chain 123 members only
    "hg":     lambda e: bool(e & {"Tl", "Hg"}),                        # Tl/Hg-12(n-1)n only
    "bi":     lambda e: True,                                          # Tl-22(n-1)n (all matches)
    "tprime": lambda e: "Ce" in e or (bool(e & {"Nd", "Pr", "Sm", "Eu"}) and not (e & {"Sr", "Ba"})),
}
MATCHER = StructureMatcher(ltol=0.1, stol=0.15, angle_tol=3, primitive_cell=True,
                           scale=True, attempt_supercell=False, comparator=FrameworkComparator())


def framework(cif_path):
    s = Structure.from_file(cif_path)
    # synth-doped CIFs carry partial occupancies: a site is a cation site if any
    # of its species is not an anion (occupancy is irrelevant to the framework)
    keep = [site for site in s if any(el.symbol not in ANIONS for el in site.species)]
    fw = Structure(Lattice(s.lattice.matrix), ["Na"] * len(keep), [site.frac_coords for site in keep])
    return fw.get_primitive_structure()


def mpid(cid):
    m = re.search(r"(mp-\d+)", cid)
    return m.group(1) if m else cid


def main(fams):
    idx = pd.read_pickle(INDEX)
    ic = [c for c in idx.columns if c in ("id", "cif_id", "name")][0]
    ids = idx[ic].astype(str).tolist()
    parents = {}
    for cid in ids:
        parents.setdefault(mpid(cid), cid)
    print(f"{len(ids)} rows, {len(parents)} unique MP parents — building frameworks", flush=True)
    fw = {}
    for k, (p, cid) in enumerate(parents.items()):
        try:
            fw[p] = framework(f"{MP}/cifs/{cid}")
        except Exception as e:  # noqa: BLE001
            print(f"  framework fail {p} ({cid[:40]}): {str(e)[:60]}", flush=True)
        if (k + 1) % 300 == 0:
            print(f"  {k + 1}/{len(parents)}", flush=True)

    def same(a, b):
        if a is None or b is None or len(a) != len(b):
            return False
        if abs(a.volume / len(a) - b.volume / len(b)) / (b.volume / len(b)) > 0.15:
            return False
        return MATCHER.fit(a, b)

    for fam in fams:
        hold = set(pd.read_csv(f"{MP}/holdout_fam_{fam}_ids.csv")["id"].astype(str))
        prot_parents = PROTECT_PARENT.get(fam) or sorted({mpid(i) for i in hold})
        prot = []
        for p in prot_parents:
            if p in fw and not any(same(fw[p], q) for q in prot):
                prot.append(fw[p])
        match_parent = {p: any(same(f, q) for q in prot) for p, f in fw.items()}
        idx["match"] = idx[ic].astype(str).map(lambda i: match_parent.get(mpid(i), False))
        sus = set(pd.read_csv(SUSPECTS[fam])["id"].astype(str)) if fam in SUSPECTS else set()
        chem = CHEM.get(fam, lambda e: True)
        elems = lambda i: set(re.findall(r"[A-Z][a-z]?", i.split("-MP-")[0].split("-ICSD-")[0]))
        mt = dict(zip(idx[ic].astype(str), idx["match"]))
        new_hold = [i for i in ids if i in hold and mt[i] and i not in sus]
        excl = [i for i in ids if i not in new_hold and ((mt[i] and chem(elems(i))) or i in sus)]
        pd.DataFrame({"id": new_hold}).to_csv(f"{MP}/holdout_struct_{fam}_ids.csv", index=False)
        pd.DataFrame({"id": excl}).to_csv(f"{MP}/exclude_struct_{fam}_ids.csv", index=False)
        dropped = sorted(hold - set(new_hold) - sus)
        print(f"\n=== {fam}: {len(prot)} protected framework(s) from {prot_parents[:6]}{'...' if len(prot_parents) > 6 else ''}")
        print(f"  holdout {len(hold)} -> structure holdout {len(new_hold)} (dropped {len(dropped)} non-matching + {len(hold & sus)} suspects)")
        print(f"  exclusion: {len(excl)} rows ({len(excl) - len(set(excl) & sus)} framework matches outside the holdout + {len(set(excl) & sus)} suspects)")
        ex_f = [i.split("-MP-")[0].split("-ICSD-")[0] for i in excl if i not in sus]
        fams_el = pd.Series([re.findall(r"[A-Z][a-z]?", f) for f in ex_f]).explode().value_counts()
        print("  excluded-row elements:", {k: int(v) for k, v in fams_el.head(14).items()})
        print("  excluded examples:", ", ".join(ex_f[:10]))
        if dropped:
            print("  dropped from holdout (different framework):", ", ".join(i.split("-MP-")[0] for i in dropped[:8]), "...")


if __name__ == "__main__":
    main(sys.argv[1:] or ["yba", "bi", "hg", "tprime"])
