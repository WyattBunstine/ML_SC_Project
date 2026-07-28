"""Full v4.3 rebuild of the DOS-member graphs (dos_all_index) with baked CF.

Why: the DOS member's graphs lived in the shared graphs_v4 dir across builder
eras — 39,218 of 62,478 predate the positions/to_jimage schema, which broke the
augment-cf backfill (fail:no-positions) and produced the corrupted mixed pack
dos_pack_ef1_cf (63% zero-filled rows served as baked). A full rebuild from
structures is the only path that guarantees uniform v4.3 graphs.

Two phases (run separately so the API fetch can overlap other work):
  --fetch   complete the local CIF set: fetch missing structures from MP
            (MP_API_KEY env) and write them into dos_rebuild/cifs/ so the
            source set becomes 62,478 self-contained CIFs (no future re-fetch).
            NOTE: re-fetched structures are TODAY's MP calcs; the attached DOS
            arrays come from the original fetch — same material ground state,
            possibly a newer relaxation. Acceptable for a per-structure total
            DOS target; recorded here for provenance.
  --build   build every graph with the CURRENT builder (schema >= v4.3, baked
            valence+cf), carry the per-structure physics keys (dos, bandgap,
            forces, magmom, stress) over from the OLD graph JSON, write to
            dos_rebuild/graphs_v4_cf/, and emit dos_all_index_cf.pickle/.csv.
            Refuses to finish silently on failures: writes failed_paths.txt and
            exits nonzero if ANY graph failed (no sub-1% tolerance — the DOS
            corruption taught us how "1%" compounds through any-graph flags).

  MP_API_KEY=... python scripts/rebuild_dos_graphs_cf.py --fetch
  python scripts/rebuild_dos_graphs_cf.py --build [--workers N]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "database"))

DOS_DIR = "database/datafiles/MP/dos_rebuild"
CIF_DIR = os.path.join(DOS_DIR, "cifs")
OUT_GRAPHS = os.path.join(DOS_DIR, "graphs_v4_cf")
INDEX_IN = os.path.join(DOS_DIR, "dos_all_index.pickle")
INDEX_OUT = os.path.join(DOS_DIR, "dos_all_index_cf")
PHYS_KEYS = ("dos", "bandgap", "forces", "magmom", "stress")


def cif_name(mp_id: str) -> str:
    return mp_id if mp_id.endswith(".cif") else mp_id + ".cif"


def fetch_missing():
    key = os.environ.get("MP_API_KEY") or sys.exit("set MP_API_KEY")
    from mp_api.client import MPRester
    from pymatgen.io.cif import CifWriter
    df = pd.read_pickle(INDEX_IN)
    have = set(os.listdir(CIF_DIR))
    missing = [str(i) for i in df["id"] if cif_name(str(i)) not in have]
    print(f"{len(missing):,} structures to fetch ({len(df) - len(missing):,} local)", flush=True)
    fetched = failed = 0
    CH = 500
    with MPRester(key) as mpr:
        for i in range(0, len(missing), CH):
            chunk = missing[i:i + CH]
            docs = mpr.materials.summary.search(material_ids=chunk, fields=["material_id", "structure"])
            got = {str(d.material_id): d.structure for d in docs}
            for mid in chunk:
                st = got.get(mid)
                if st is None:
                    failed += 1
                    continue
                try:
                    CifWriter(st, write_magmoms=False).write_file(os.path.join(CIF_DIR, cif_name(mid)))
                    fetched += 1
                except Exception:  # noqa: BLE001
                    failed += 1
            print(f"  {min(i + CH, len(missing)):,}/{len(missing):,} fetched={fetched:,} "
                  f"unavailable={failed:,}", flush=True)
    print(f"FETCH DONE: {fetched:,} written, {failed:,} unavailable "
          f"(deprecated MP ids get dropped from the new index at --build)", flush=True)


def _build_one(task):
    """(mp_id, cif_path, old_graph_path, out_path) -> 'done' | 'fail:<why>'."""
    mp_id, cif_path, old_path, out_path = task
    try:
        from pymatgen.core import Structure
        import crystal_graph_v4_import  # noqa: F401
        from crystal_graph_v4 import build_crystal_graph_from_structure
        from database.database_main import _compact_v4_graph
        st = Structure.from_file(cif_path)
        g = _compact_v4_graph(build_crystal_graph_from_structure(st))
        with open(old_path) as f:
            old = json.load(f)
        for k in PHYS_KEYS:
            if k in old:
                g[k] = old[k]
        if "dos" not in g:
            return "fail:old-graph-missing-dos"
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(g, f)
        os.replace(tmp, out_path)
        return "done"
    except Exception as exc:  # noqa: BLE001
        return f"fail:{type(exc).__name__}"


def build_all(workers=None):
    df = pd.read_pickle(INDEX_IN)
    os.makedirs(OUT_GRAPHS, exist_ok=True)
    tasks, drop_nocif = [], []
    for _, row in df.iterrows():
        mid = str(row["id"])
        cif = os.path.join(CIF_DIR, cif_name(mid))
        out = os.path.join(OUT_GRAPHS, cif_name(mid) + ".json")
        if not os.path.exists(cif):
            drop_nocif.append(mid)
            continue
        if os.path.exists(out):        # resumable: completed graphs skipped
            continue
        tasks.append((mid, cif, row["graph_path"], out))
    print(f"{len(tasks):,} graphs to build ({len(drop_nocif):,} no-cif dropped, "
          f"{len(df) - len(tasks) - len(drop_nocif):,} already built)", flush=True)

    import multiprocessing as mp
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    fails = {}
    failed_paths = []
    t0 = time.time()
    done = 0
    with mp.Pool(workers) as pool:
        for (task, res) in zip(tasks, pool.imap(_build_one, tasks, chunksize=8)):
            if res == "done":
                done += 1
            else:
                fails[res] = fails.get(res, 0) + 1
                failed_paths.append(f"{task[0]}\t{res}")
            n = done + len(failed_paths)
            if n % 2000 == 0:
                print(f"  {n:,}/{len(tasks):,} ({n / (time.time() - t0):.0f}/s) "
                      f"fail={len(failed_paths)}", flush=True)
    if failed_paths:
        with open(os.path.join(OUT_GRAPHS, "failed_paths.txt"), "w") as f:
            f.write("\n".join(failed_paths) + "\n")

    # new index: only rows whose NEW graph exists (dropped: no-cif + failed)
    keep_rows = []
    for _, row in df.iterrows():
        out = os.path.join(OUT_GRAPHS, cif_name(str(row["id"])) + ".json")
        if os.path.exists(out):
            r = dict(row)
            r["graph_path"] = out
            keep_rows.append(r)
    out_df = pd.DataFrame(keep_rows)
    out_df.to_pickle(INDEX_OUT + ".pickle")
    out_df.to_csv(INDEX_OUT + ".csv", index=False)
    print(f"BUILD DONE in {time.time() - t0:.0f}s: {done:,} built this run, "
          f"index {len(out_df):,} rows -> {INDEX_OUT}.pickle "
          f"(dropped {len(drop_nocif):,} no-cif, {len(failed_paths)} failed "
          f"{dict(list(fails.items())[:4]) if fails else ''})", flush=True)
    if failed_paths:
        sys.exit(f"{len(failed_paths)} build failures — see {OUT_GRAPHS}/failed_paths.txt; "
                 "index excludes them, but investigate before packing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    if a.fetch:
        fetch_missing()
    if a.build:
        build_all(a.workers)
    if not (a.fetch or a.build):
        sys.exit("pass --fetch and/or --build")
