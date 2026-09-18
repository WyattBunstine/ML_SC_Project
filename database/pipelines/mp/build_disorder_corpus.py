"""Disorder-corpus generator: ordered mixed-element MP structures -> fractional-
occupancy 'doped' structures with the ordered structure's DFT energy inherited.

Principle (mean-field learning): an ordered arrangement is ONE sample of the
disordered ensemble; the occupancy-weighted encoder can't resolve orderings, so
training it against ordered samples teaches the ensemble-average energy. No DFT.

Criterion: a geometrically-equivalent site orbit (symprec 0.3, species-blind) that
holds >=2 DIFFERENT elements which pass the Hume-Rothery solid-solution filter
(metallic/atomic radius diff < 15% AND |dEN| < 0.8). Spans cation-site (LSCO),
anion-site (S-Se, the FeSe family), and alloy-site (Nb-Ti/A15) mixing.

Writes disordered CIFs + a source CSV (cif, formation_energy_per_atom) that
`main.py build-db --kind cgv4` turns into occupancy-weighted graphs.

CIFs are named bare {mp_id}.cif ON PURPOSE: the pack's derive_mp_id groups come
from the filename stem, so a material's disordered variant lands in the SAME
material-split group as its ordered MPtrj trajectory (correlated samples stay in
one fold). The cost of that choice: the graph JSONs share the mp-XXXX.cif.json
namespace with ordered builds, and build-db's resume-by-existence would silently
reuse ordered graphs from a shared graph dir. Therefore the graph build MUST
target a dedicated dir under --out-dir (the printed instruction), and this script
refuses to write into a non-empty cifs/ unless --fresh wipes it.

  MP_API_KEY=... python database/pipelines/mp/build_disorder_corpus.py --limit 1500 --out-dir <dir>
"""
import argparse
import csv
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter


def _radius(el):
    e = Element(el)
    for r in (e.metallic_radius, e.atomic_radius, e.average_ionic_radius):
        try:
            if r:
                return float(r)
        except Exception:  # noqa: BLE001
            pass
    return 1.5


def hume_rothery(a, b, r_tol=0.15, en_tol=0.8):
    ra, rb = _radius(a), _radius(b)
    rd = abs(ra - rb) / max(ra, rb)
    en = abs(float(Element(a).X or 0) - float(Element(b).X or 0))
    return rd < r_tol and en < en_tol


def disorder_structure(st, symprec=0.3):
    """Merge every Hume-Rothery-compatible mixed-element orbit into fractional
    occupancy. Returns (disordered Structure, n_orbits_merged, pairs, spacegroup#)
    or (None, 0, [], 0)."""
    dummy = Structure(st.lattice, ["C"] * len(st), st.frac_coords)
    try:
        sga = SpacegroupAnalyzer(dummy, symprec=symprec)
        sgnum = sga.get_space_group_number()
        sym = sga.get_symmetrized_structure()
    except Exception:  # noqa: BLE001
        return None, 0, [], 0
    site_species = [{Element(st[i].specie.symbol): 1.0} for i in range(len(st))]
    merged, pairs = 0, []
    for orbit in sym.equivalent_indices:
        els = [st[i].specie.symbol for i in orbit]
        distinct = sorted(set(els))
        if len(distinct) < 2:
            continue
        if not all(hume_rothery(distinct[i], distinct[j])
                   for i in range(len(distinct)) for j in range(i + 1, len(distinct))):
            continue
        cnt = Counter(els)
        avg = {Element(e): c / len(orbit) for e, c in cnt.items()}   # mean-field occupancy
        for i in orbit:
            site_species[i] = avg
        merged += 1
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                pairs.append((distinct[i], distinct[j]))
    if merged == 0:
        return None, 0, [], 0
    dis = Structure(st.lattice, [site_species[i] for i in range(len(st))], st.frac_coords)
    return dis, merged, pairs, sgnum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500, help="MP candidates to scan")
    ap.add_argument("--out-dir", default="database/datafiles/MP/disorder_corpus")
    ap.add_argument("--e-hull-max", type=float, default=0.5)
    ap.add_argument("--symprec", type=float, default=0.3)
    ap.add_argument("--fresh", action="store_true",
                    help="wipe an existing cifs/ + source.csv before building")
    args = ap.parse_args()
    key = os.environ.get("MP_API_KEY") or sys.exit("set MP_API_KEY")
    from mp_api.client import MPRester
    cif_dir = os.path.join(args.out_dir, "cifs")
    csv_path = os.path.join(args.out_dir, "source.csv")
    # A non-empty cifs/ from an earlier (possibly partial/different-seed) run would
    # leave orphans that disagree with source.csv — refuse unless --fresh wipes it.
    if os.path.isdir(cif_dir) and os.listdir(cif_dir):
        if not args.fresh:
            sys.exit(f"{cif_dir} is not empty (a prior run?). Pass --fresh to wipe "
                     "and rebuild, or use a different --out-dir.")
        import shutil
        shutil.rmtree(cif_dir)
        if os.path.exists(csv_path):
            os.remove(csv_path)
    os.makedirs(cif_dir, exist_ok=True)

    import random
    with MPRester(key) as mpr:
        pool = [str(d.material_id) for d in mpr.materials.summary.search(
            num_elements=(2, 4), energy_above_hull=(0, args.e_hull_max), fields=["material_id"])]
        # The API's result order is not guaranteed — sort before the seeded sample
        # so the same seed + limit reproduces the same candidate set.
        pool.sort()
        print(f"MP pool (2-4 elem, e_hull<{args.e_hull_max}): {len(pool):,}", flush=True)
        random.seed(0)
        cand = random.sample(pool, min(args.limit, len(pool)))
        pair_tally, done, dup_skipped = Counter(), 0, 0
        # Dedup alternate orderings of the SAME solid solution: distinct MP entries
        # that orbit-average to near-identical disordered structures (same reduced
        # composition + spacegroup + volume/atom) carry correlated energies under
        # different mp_ids — the material split would scatter them across folds,
        # leak-inflating the disorder member's val loss. Keep the first.
        seen_solutions = set()
        CH = 500
        # source.csv written INCREMENTALLY (one row per CIF as it lands, flushed per
        # chunk) so a mid-run failure leaves cifs/ and source.csv consistent instead
        # of orphaned CIFs beside an empty CSV.
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cif", "formation_energy_per_atom"])
            for i in range(0, len(cand), CH):
                docs = mpr.materials.summary.search(
                    material_ids=cand[i:i + CH],
                    fields=["material_id", "structure", "formation_energy_per_atom"])
                for d in docs:
                    st = d.structure
                    if not st.is_ordered or d.formation_energy_per_atom is None:
                        continue
                    dis, n_orb, pairs, sgnum = disorder_structure(st, args.symprec)
                    if dis is None:
                        continue
                    sol_key = (dis.composition.reduced_formula, sgnum,
                               round(dis.volume / max(len(dis), 1), 1))
                    if sol_key in seen_solutions:
                        dup_skipped += 1
                        continue
                    mid = str(d.material_id)
                    cif_path = os.path.join(cif_dir, f"{mid}.cif")
                    try:
                        CifWriter(dis, write_magmoms=False).write_file(cif_path)
                    except Exception:  # noqa: BLE001
                        continue
                    seen_solutions.add(sol_key)
                    w.writerow([f"{mid}.cif", float(d.formation_energy_per_atom)])
                    pair_tally.update(pairs)
                    done += 1
                f.flush()
                print(f"  scanned {min(i + CH, len(cand))}/{len(cand)} -> {done} disordered "
                      f"({dup_skipped} ordering-duplicates skipped)", flush=True)
    print(f"\nDONE: {done} disordered structures -> {cif_dir} "
          f"({dup_skipped} alternate-ordering duplicates skipped)")
    print(f"  source.csv (cif, formation_energy_per_atom) written")
    print("  top solid-solution pairs:",
          {f"{a}-{b}": v for (a, b), v in pair_tally.most_common(10)})
    print(f"\n  NEXT: build graphs into a DEDICATED dir, e.g.\n"
          f"    python main.py build-db --kind cgv4 --source {csv_path} \\\n"
          f"        --graph-dir {os.path.join(args.out_dir, 'graphs_v4')}\n"
          f"  (bare mp-XXXX.cif names share the ordered builds' graph namespace — a\n"
          f"   shared graph dir would silently reuse ORDERED graphs on resume.)")


if __name__ == "__main__":
    main()
