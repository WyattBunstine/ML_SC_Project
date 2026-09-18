"""Fetch MP DFPT phonon DOS (Petretto set, ~1.5k materials incl. oxides) — task #8.

Discovers which materials carry phonon data (summary has_props), fetches each
PhononDos (frequencies THz + densities), and caches them raw to
database/datafiles/MP_PhononDOS/raw/<mp_id>.npz — resumable (skips existing).
Binning onto the shared PHONON grid + graph baking is build_phonon_targets.py's
job; this script only banks the raw data.

  MP_API_KEY=... python database/pipelines/phonon/fetch_mp_phonon_dos.py [--limit N]
"""
import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root from database/pipelines/<group>/
OUT = os.path.join(_ROOT, "database", "datafiles", "MP_PhononDOS", "raw")


def main(limit=None):
    from mp_api.client import MPRester
    os.makedirs(OUT, exist_ok=True)
    key = os.environ.get("MP_API_KEY")
    if not key:
        sys.exit("MP_API_KEY not set")
    with MPRester(key) as m:
        docs = m.materials.summary.search(has_props=["phonon"],
                                          fields=["material_id"])
        ids = sorted({str(d.material_id) for d in docs})
        print(f"{len(ids)} materials advertise phonon data "
              f"(dfpt 1,513 + pheasy 26,237 docs)", flush=True)
        if limit:
            ids = ids[:limit]
        n_ok = n_skip = n_fail = 0
        for i, mid in enumerate(ids):
            dest = os.path.join(OUT, f"{mid}.npz")
            if os.path.exists(dest):
                n_skip += 1
                continue
            # Method priority: dfpt (curated Petretto set) over pheasy (the 2025
            # 26k expansion). The method is stored so pack-time source selection
            # can rank sources (dfpt > togo > pheasy) explicitly.
            dos = method = None
            for meth in ("dfpt", "pheasy"):
                try:
                    dos = m.materials.phonon.get_dos_from_material_id(mid, meth)
                    method = meth
                    break
                except Exception:  # noqa: BLE001 — try the next method
                    continue
            if dos is None:
                n_fail += 1
                if n_fail <= 20 or n_fail % 100 == 0:
                    print(f"fail {mid}: no dos under any method", flush=True)
            else:
                np.savez_compressed(dest + ".tmp.npz",
                                    frequencies=np.asarray(dos.frequencies, dtype=np.float64),
                                    densities=np.asarray(dos.densities, dtype=np.float64),
                                    method=np.array(method))
                os.replace(dest + ".tmp.npz", dest)
                n_ok += 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(ids)} (ok {n_ok}, skip {n_skip}, fail {n_fail})",
                      flush=True)
        print(f"fetch done: ok {n_ok}, skip {n_skip}, fail {n_fail}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    main(ap.parse_args().limit)
