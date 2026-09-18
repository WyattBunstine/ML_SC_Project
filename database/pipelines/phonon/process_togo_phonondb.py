"""Compute total phonon DOS from the crawled Togo phonondb zips (task #9).

Per zip: extract phonopy_params.yaml.xz -> phonopy.load (force constants) ->
20^3 mesh -> total DOS on a fine pitch -> cache RAW curve + the 256-bin
PHONON-grid vector to database/datafiles/TogoPhononDB/dos_raw/<mp_id>.npz.
Resumable (skips existing); multiprocessing. Pack assembly / graph baking is a
separate step (source dedup vs MP dfpt/pheasy happens there).

  python database/pipelines/phonon/process_togo_phonondb.py [--workers 4] [--limit N]

Memory: phonopy's Mesh.run() builds the dynamical matrices of ALL irreducible
q-points as one (n_q, 3N, 3N) complex array — 15 GiB for a 168-atom cell at
20^3 (4,000 ir q-points under the time-reversal-only fallback). That is what
OOM-killed the 2026-09-08 runs (and took the IDE session down with them).
Cells above BIG_ATOMS therefore go through _mesh_total_dos's chunked path
(same ir q-points, same solver, same smearing — bounded memory), and every
worker runs under an RLIMIT_DATA so any surprise surfaces as a clean per-material
"fail" instead of a kernel OOM kill.
"""
import os

# BLAS/rayon threads per worker — set before numpy/phonopy import.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("RAYON_NUM_THREADS", "2")

import argparse  # noqa: E402
import glob  # noqa: E402
import io  # noqa: E402
import lzma  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root from database/pipelines/<group>/
sys.path.insert(0, os.path.join(_ROOT, "models", "common"))
from data import PHONON_N_BINS, PHONON_W_MAX_THZ, bin_spectrum  # noqa: E402

TOGO = os.path.join(_ROOT, "database", "datafiles", "TogoPhononDB")
OUT = os.path.join(TOGO, "dos_raw")
SIGMA_THZ = 0.1
FREQ_PITCH_THZ = 0.02
MESH = [20, 20, 20]
BIG_ATOMS = 40  # batched Mesh.run() peak ~ 4000 * (3N)^2 * 16 B = 0.9 GiB at N=40
CHUNK_BYTES = 4e8  # dynamical-matrix batch budget per run_qpoints call
WORKER_AS_BYTES = int(float(os.environ.get("TOGO_WORKER_AS_GB", "5")) * 2**30)
FORCE_CHUNKED = os.environ.get("TOGO_FORCE_CHUNKED") == "1"  # validation only


def _mesh_total_dos(ph):
    """(frequency_points, dos) on MESH, Gaussian SIGMA_THZ, FREQ_PITCH_THZ.

    Small cells: phonopy's own run_mesh/run_total_dos. Big cells: the same
    irreducible q-points and weights from a Mesh that is never .run(), the
    frequencies solved in bounded chunks via run_qpoints (same solver, same
    unit factor), and the same TotalDos class smearing them.
    """
    n_atoms = len(ph.primitive)
    if n_atoms <= BIG_ATOMS and not FORCE_CHUNKED:
        ph.run_mesh(MESH)
        ph.run_total_dos(sigma=SIGMA_THZ, freq_pitch=FREQ_PITCH_THZ)
        td = ph.total_dos
        return td.frequency_points, td.dos

    from phonopy.phonon.dos import TotalDos
    from phonopy.phonon.mesh import Mesh

    grid = Mesh(ph.dynamical_matrix, MESH,
                rotations=ph.primitive_symmetry.pointgroup_operations,
                primitive_symmetry=ph.primitive_symmetry,
                factor=ph.unit_conversion_factor)  # ir q-points + weights only
    nb = 3 * n_atoms
    chunk = max(1, int(CHUNK_BYTES // (16 * nb * nb)))
    freqs = []
    for i in range(0, len(grid.qpoints), chunk):
        ph.run_qpoints(grid.qpoints[i:i + chunk])
        freqs.append(np.array(ph.qpoints.frequencies, dtype=np.float64))

    class _Solved:  # duck-typed stand-in for a run Mesh
        frequencies = np.concatenate(freqs)
        weights = grid.weights
        primitive = ph.primitive

    td = TotalDos(_Solved(), sigma=SIGMA_THZ)
    td.set_draw_area(freq_pitch=FREQ_PITCH_THZ)
    td.run()
    return td.frequency_points, td.dos


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
                f_thz, d = _mesh_total_dos(ph)
        finally:
            os.unlink(tmp)
        f_thz = np.asarray(f_thz, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        n_prim = len(ph.primitive)
        total = np.trapezoid(d, f_thz)
        pos = f_thz > 0
        if total <= 0 or not pos.any():
            # zero/degenerate DOS (broken force constants, all-imaginary modes):
            # caching it as "ok" would poison the pack forever via skip-existing.
            return f"fail {mp_id}: degenerate DOS (total integral {total:.3g})"
        frac_pos = np.trapezoid(d[pos], f_thz[pos]) / total
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
        return f"fail {mp_id}: {type(e).__name__}: {str(e)[:100]}"


def _init_worker():
    """Hard per-worker anonymous-memory cap: a runaway allocation raises
    MemoryError inside process_one's try/except instead of inviting the
    kernel OOM killer (which took down the whole IDE cgroup on 2026-09-08)."""
    import resource
    # RLIMIT_DATA (heap + private anonymous mmaps, i.e. numpy arrays), not
    # RLIMIT_AS: the torch/CUDA libraries data.py pulls in map gigabytes of
    # virtual address space that never becomes resident.
    resource.setrlimit(resource.RLIMIT_DATA, (WORKER_AS_BYTES, WORKER_AS_BYTES))


if __name__ == "__main__":
    from multiprocessing import Pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    zips = sorted(glob.glob(os.path.join(TOGO, "zips", "*.zip")))
    if a.limit:
        zips = zips[: a.limit]
    n = {"ok": 0, "skip": 0, "fail": 0}
    print(f"{len(zips)} zips, {a.workers} workers, RLIMIT_DATA "
          f"{WORKER_AS_BYTES / 2**30:.1f} GiB/worker, chunked above {BIG_ATOMS} atoms",
          flush=True)
    # no maxtasksperchild: with it, 3 of 4 workers sat blocked on the inqueue
    # lock after the first recycle (2026-09-08) — the plain pool never stalled.
    with Pool(a.workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(process_one, zips, chunksize=4)):
            n["fail" if res.startswith("fail") else res] += 1
            if res.startswith("fail") and n["fail"] <= 100:
                print(res, flush=True)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(zips)} (ok {n['ok']}, skip {n['skip']}, "
                      f"fail {n['fail']})", flush=True)
    print(f"togo dos done: {n}", flush=True)
