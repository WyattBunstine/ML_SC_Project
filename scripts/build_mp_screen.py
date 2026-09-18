"""Assemble the MP discovery-screen inputs for predict_tc.py (user, 2026-09-09):
every MP material that already has a graph in the v45 encoder schema locally —
the phonon-corpus graphs (31k) and the DOS-rebuild graphs (62k), deduped by
mp-id (phonon graph preferred) — as an index pickle + a 41-dim descriptor
table, plus MP summary metadata (formula, e_above_hull, band gap, theoretical
flag) for ranking and filtering.
  python scripts/build_mp_screen.py index      -> database/datafiles/MP/screen_mp_index.pickle + descriptors_screen_mp.pickle
  python scripts/build_mp_screen.py metadata   -> database/datafiles/MP/screen_mp_metadata.pickle
"""
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
MP = os.path.join(_ROOT, "database", "datafiles", "MP")
INDEX = os.path.join(MP, "screen_mp_index.pickle")
DESC = os.path.join(MP, "descriptors_screen_mp.pickle")
META = os.path.join(MP, "screen_mp_metadata.pickle")
SOURCES = [os.path.join(_ROOT, "database", "datafiles", "MP_PhononDOS", "graphs_v45_phdos"),
           os.path.join(MP, "dos_rebuild", "graphs_v4_cf")]


def cmd_index():
    rows, seen = [], set()
    for src in SOURCES:
        for f in sorted(os.listdir(src)):
            if not f.endswith(".json"):
                continue
            mid = f[:-5]
            if mid.endswith(".cif"):          # dos_rebuild graphs are named mp-XXXX.cif.json
                mid = mid[:-4]
            if not mid.startswith("mp-") or mid in seen:
                continue
            seen.add(mid)
            rows.append({"id": mid, "value": 0.0, "graph_path": os.path.relpath(os.path.join(src, f), _ROOT),
                         "label": 0, "mp_id": mid})
    df = pd.DataFrame(rows)
    df.to_pickle(INDEX)
    print(f"index: {len(df)} MP materials -> {INDEX}", flush=True)
    sys.path.insert(0, os.path.join(_ROOT, "models", "head"))
    from descriptors import build_descriptor_table
    build_descriptor_table(INDEX, DESC)
    d = pd.read_pickle(DESC)
    print(f"descriptors: {len(d['table'])} rows, {len(d['names'])} dims, {len(d.get('failed', []))} failed -> {DESC}", flush=True)


def cmd_metadata():
    from mp_api.client import MPRester
    ids = sorted({i[:-4] if i.endswith(".cif") else i for i in pd.read_pickle(INDEX)["id"]})
    out = []
    if os.path.exists(META):                  # resumable: keep what an earlier pass fetched
        prev = pd.read_pickle(META)
        out = prev.to_dict("records")
        have = set(prev["id"])
        ids = [i for i in ids if i not in have]
    print(f"metadata: {len(ids)} ids to fetch", flush=True)
    with MPRester(os.environ["MP_API_KEY"]) as m:
        for i in range(0, len(ids), 1000):
            chunk = ids[i:i + 1000]
            try:
                docs = m.materials.summary.search(
                    material_ids=chunk, fields=["material_id", "formula_pretty", "energy_above_hull", "band_gap",
                                                "is_metal", "theoretical", "nsites", "symmetry", "is_magnetic",
                                                "total_magnetization", "density"])
            except Exception as e:  # noqa: BLE001
                print(f"  chunk {i}: {str(e)[:80]}", flush=True)
                continue
            for d in docs:
                out.append({"id": str(d.material_id), "formula": d.formula_pretty, "e_above_hull": d.energy_above_hull,
                            "band_gap": d.band_gap, "is_metal": d.is_metal, "theoretical": d.theoretical, "nsites": d.nsites,
                            "spacegroup": getattr(d.symmetry, "symbol", None), "crystal_system": str(getattr(d.symmetry, "crystal_system", "")),
                            "is_magnetic": d.is_magnetic, "magnetization": d.total_magnetization, "density": d.density})
            if (i // 1000) % 10 == 0:
                print(f"  metadata {i + len(chunk)}/{len(ids)}", flush=True)
    df = pd.DataFrame(out).drop_duplicates("id")
    df.to_pickle(META)
    print(f"metadata: {len(df)} rows -> {META}", flush=True)


if __name__ == "__main__":
    {"index": cmd_index, "metadata": cmd_metadata}[sys.argv[1]]()
