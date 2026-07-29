"""Backfill baked v4.3 electronic-structure blocks (valence + AOM cf) onto
EXISTING compact graph JSONs — the cheap alternative to a full rebuild for
ORDERED datasets (MPtrj frames, relaxed MP/DOS structures).

The expensive builder stages (Voronoi tessellation, ECoN, oxidation guess) are
unchanged in v4.3; only the new per-node blocks are missing. Compact graphs
carry everything the AOM needs: per-edge to_jimage + directional ECoN weights,
per-node ion_role/Z/chi/oxidation, and graph-level frac_coords + lattice
(augment-positions backfill), from which exact cartesian bond vectors are
reconstructed as (frac_tgt + jimage - frac_src) @ lattice.

EXACTNESS: for ordered sites this reproduces the full v4.3 builder bit-for-bit
(validated against the SC full rebuild — see task #19 notes). Compact graphs
do NOT carry species lists, so fractionally-occupied sites would fall back to
dominant-element + site-averaged oxidation; datasets with mixed sites (the SC
doped set, the disorder corpus) must use the full rebuild instead. MPtrj and
the DOS set are 100% ordered.

Resumable: graphs whose first node already has a "cf" key are skipped.
Atomic: writes via tmp + os.replace so a killed run never corrupts a graph.

  python scripts/augment_cf.py --graph-dir <dir> [--workers N] [--limit N]
  python scripts/augment_cf.py --index <index.pickle>   # augment referenced graphs
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "database"))
import crystal_graph_v4_import  # noqa: F401,E402  (RPToleranceFactor on path)
from crystal_field_aom import CF_SCHEMA, site_cf_features, site_valence_subshells  # noqa: E402
from bond_valence import site_bvs_mixed  # noqa: E402

FORCE = False


def _init_force(force):
    global FORCE
    FORCE = force


# compact ion_role encoding (database_main ion_role_map)
_ANION = -1

_Z_TO_SYMBOL = {}


def _symbol(z):
    s = _Z_TO_SYMBOL.get(z)
    if s is None:
        from pymatgen.core import Element
        s = Element.from_Z(int(z)).symbol
        _Z_TO_SYMBOL[z] = s
    return s


def augment_graph(path):
    """Returns 'done', 'skip' (already augmented), or 'fail:<reason>'."""
    try:
        with open(path) as f:
            g = json.load(f)
        nodes = g.get("nodes") or []
        if not nodes:
            return "fail:no-nodes"
        if "cf" in nodes[0] and g.get("cf_schema") == CF_SCHEMA and not FORCE:
            return "skip"
        frac = g.get("frac_coords")
        lat = g.get("lattice")
        if frac is None or lat is None:
            return "fail:no-positions"
        frac = np.asarray(frac, dtype=float)
        lat = np.asarray(lat, dtype=float).reshape(3, 3)

        # per-node anion-gated neighbor gather, mirroring the v4.3 builder:
        # cart_vec points AWAY from the center; ECoN weight from the center's
        # own perspective; only anion-role neighbors exert a ligand field.
        nbrs = [[] for _ in nodes]
        bvs_nbrs = [[] for _ in nodes]
        for e in g.get("edges", []):
            s, t = int(e["source"]), int(e["target"])
            img = np.asarray(e.get("to_jimage") or (0, 0, 0), dtype=float)
            vec = (frac[t] + img - frac[s]) @ lat
            if nodes[t]["ion_role"] == _ANION:
                nbrs[s].append({"vec": vec, "weight": float(e["ecn_weight_src"]),
                                "ligand": _symbol(nodes[t]["Z"]),
                                "chi": nodes[t].get("chi_pauling")})
            if nodes[s]["ion_role"] == _ANION:
                nbrs[t].append({"vec": -vec, "weight": float(e["ecn_weight_tgt"]),
                                "ligand": _symbol(nodes[s]["Z"]),
                                "chi": nodes[s].get("chi_pauling")})
            if nodes[s]["ion_role"] != nodes[t]["ion_role"]:
                d = float(np.linalg.norm(vec))
                bvs_nbrs[s].append({"dist": d, "ligand": _symbol(nodes[t]["Z"])})
                bvs_nbrs[t].append({"dist": d, "ligand": _symbol(nodes[s]["Z"])})

        g["cf_schema"] = CF_SCHEMA
        for i, n in enumerate(nodes):
            oxi = float(n.get("oxidation_state") or 0.0)
            species = [{"symbol": _symbol(n["Z"]), "occupancy": 1.0,
                        "oxidation_state": oxi}]
            n["valence"] = site_valence_subshells(species, default_oxidation=oxi)
            cf = site_cf_features(species, nbrs[i], default_oxidation=oxi)
            n["cf"] = (cf["cf_levels"] + cf["cf_occ"]
                       + [cf["cf_frontier_gap"], cf["cf_unpaired"]])
            b = site_bvs_mixed(species, bvs_nbrs[i], default_oxidation=oxi)
            n["bvs"] = round(float(b), 6)
            n["bvs_mismatch"] = round(float(b - oxi), 6)

        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(g, f)
        os.replace(tmp, path)
        return "done"
    except Exception as exc:  # noqa: BLE001
        return f"fail:{type(exc).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir")
    ap.add_argument("--index", help="index pickle; augment every referenced graph")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true",
                    help="re-bake even if cf present (stale cf_schema re-bakes always)")
    args = ap.parse_args()
    if bool(args.graph_dir) == bool(args.index):
        sys.exit("pass exactly one of --graph-dir / --index")
    if args.index:
        import pandas as pd
        df = pd.read_pickle(args.index)
        paths = sorted(set(df["graph_path"].tolist()))
    else:
        paths = sorted(glob.glob(os.path.join(args.graph_dir, "*.json")))
    if args.limit:
        paths = paths[:args.limit]
    print(f"augmenting {len(paths):,} graphs", flush=True)

    import multiprocessing as mp
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    t0 = time.time()
    counts = {"done": 0, "skip": 0}
    fails = {}
    failed_paths = []
    with mp.Pool(workers, initializer=_init_force, initargs=(args.force,)) as pool:
        # ordered imap so results pair with paths (per-path failure manifest)
        for i, res in enumerate(pool.imap(augment_graph, paths, chunksize=64)):
            if res.startswith("fail:"):
                fails[res] = fails.get(res, 0) + 1
                failed_paths.append(f"{paths[i]}\t{res}")
            else:
                counts[res] += 1
            if (i + 1) % 20000 == 0:
                el = time.time() - t0
                print(f"  {i + 1:,}/{len(paths):,} ({(i + 1) / el:.0f}/s) "
                      f"done={counts['done']:,} skip={counts['skip']:,} "
                      f"fail={sum(fails.values())}", flush=True)
    if failed_paths:
        man = ((os.path.join(args.graph_dir, "augment_failed_paths.txt")) if args.graph_dir
               else os.path.splitext(args.index)[0] + ".augment_failed_paths.txt")
        with open(man, "w") as f:
            f.write("\n".join(failed_paths) + "\n")
        print(f"failure manifest: {man}", flush=True)
    print(f"DONE in {time.time() - t0:.0f}s: {counts['done']:,} augmented, "
          f"{counts['skip']:,} already current, {sum(fails.values())} failed "
          f"{dict(list(fails.items())[:5]) if fails else ''}", flush=True)
    if fails:
        # ANY failure blocks packing: pack-level has_* flags are pack-global, so
        # one un-baked graph becomes zeros-served-as-real (the dos_pack_ef1_cf
        # corruption). Fix the listed paths and re-run (resumable).
        sys.exit(f"{sum(fails.values())} augment failures — see manifest; "
                 "do NOT pack until clean")


if __name__ == "__main__":
    main()
