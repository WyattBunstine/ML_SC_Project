"""Compute total phonon DOS from the crawled Togo phonondb zips (task #9).

Per zip: extract phonopy_params.yaml.xz -> phonopy.load (force constants) ->
20^3 mesh -> total DOS on a fine pitch -> cache RAW curve + the 256-bin
PHONON-grid vector to database/datafiles/TogoPhononDB/dos_raw/<mp_id>.npz.
Resumable (skips existing); multiprocessing. Pack assembly / graph baking is a
separate step (source dedup vs MP dfpt/pheasy happens there).

  python scripts/process_togo_phonondb.py [--workers 8] [--limit N]
"""
import argparse
import glob
import io
import lzma
import os
import sys
import tempfile
import zipfile

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "models", "common"))
from data import PHONON_N_BINS, PHONON_W_MAX_THZ, bin_spectrum  # noqa: E402

TOGO = os.path.join(_ROOT, "database", "datafiles", "TogoPhononDB")
OUT = os.path.join(TOGO, "dos_raw")
SIGMA_THZ = 0.1


def process_one(zpath):
    mp_id = os.path.basename(zpath)[:-4]
    dest = os.path.join(OUT, f"{mp_id}.npz")
    if os.path.exists(dest):
        return "skip"
    try:
        import contextlib
        import phonopy
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.endswith("phonopy_params.yaml.xz"))
            raw = lzma.decompress(z.read(name))
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ph = phonopy.load(tmp, log_level=0)
                ph.run_mesh([20, 20, 20])
                ph.run_total_dos(sigma=SIGMA_THZ, freq_pitch=0.02)
        finally:
            os.unlink(tmp)
        f_thz = np.asarray(ph.total_dos.frequency_points, dtype=np.float64)
        d = np.asarray(ph.total_dos.dos, dtype=np.float64)
        n_prim = len(ph.primitive)
        total = np.trapezoid(d, f_thz)
        pos = f_thz > 0
        frac_pos = (np.trapezoid(d[pos], f_thz[pos]) / total) if total > 0 else 0.0
        binned = bin_spectrum(f_thz[pos], d[pos])
        dw = PHONON_W_MAX_THZ / PHONON_N_BINS
        target = 3.0 * n_prim * frac_pos
        if binned.sum() * dw > 0:
            binned = binned * (target / (binned.sum() * dw))
        np.savez_compressed(dest + ".tmp.npz", frequencies=f_thz, densities=d,
                            binned=binned.astype(np.float32), n_prim=n_prim,
                            frac_pos=frac_pos)
        os.replace(dest + ".tmp.npz", dest)
        return "ok"
    except Exception as e:  # noqa: BLE001 — per-material isolation; rerun retries
        return f"fail {mp_id}: {str(e)[:100]}"


if __name__ == "__main__":
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(TOGO, "zips", "*.zip")))
    if a.limit:
        zips = zips[: a.limit]
    n = {"ok": 0, "skip": 0, "fail": 0}
    with Pool(a.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, zips, chunksize=4)):
            n["fail" if res.startswith("fail") else res] += 1
            if res.startswith("fail") and n["fail"] <= 20:
                print(res, flush=True)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(zips)} (ok {n['ok']}, skip {n['skip']}, "
                      f"fail {n['fail']})", flush=True)
    print(f"togo dos done: {n}", flush=True)
