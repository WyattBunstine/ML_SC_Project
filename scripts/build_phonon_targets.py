"""Bake phonon-spectrum targets into graph JSONs (tasks #7/#8/#9).

Subcommands:
  a2f    — Cerqueira alpha^2F: parse a2F.dos6 (published-value smearing) from
           EPH_Cerqueira/a2f_raw/batch-*/<Formula>_<agm>/, apply the verified
           12 K low-omega cutoff, bin onto the shared PHONON grid
           (data.bin_spectrum), write g["a2f"] into graphs_v45_eph/<agm>.json.
           VALIDATES corpus-wide: lambda/omega_log recomputed from the BINNED
           spectrum vs McMillan.dat — the binning-fidelity report.
           Re-runnable (idempotent: overwrites the key).

  phdos  — Cerqueira phonon DOS from the qe.dyn* dynamical-matrix files: each
           qe.dynN (N>=1) lists its irreducible q's star multiplicity (count of
           'Dynamical  Matrix' blocks) and the frequencies ALREADY in THz
           ('freq (i) = x [THz]'); qe.dyn0 gives the full q-grid size for a
           weight sanity check. DOS = Gaussian-smeared weighted histogram on
           the shared PHONON grid, normalized so the integral is 3*nat
           (extensive — _assemble_targets divides by n_atoms). Imaginary
           (negative) modes are dropped with their weight reported.
           Writes g["ph_dos"] into graphs_v45_eph/<agm>.json. Re-runnable.

  python scripts/build_phonon_targets.py a2f|phdos [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "models", "common"))
from data import PHONON_N_BINS, PHONON_W_MAX_THZ, bin_spectrum  # noqa: E402

EPH = os.path.join(_ROOT, "database", "datafiles", "EPH_Cerqueira")
RY_TO_K = 13.605693122994 / 8.617333262e-5
RY_TO_THZ = 13605.693122994 / 4.135667696
THZ_TO_K = RY_TO_K / RY_TO_THZ
CUT_K = 12.0                     # soft-mode smearing tail (verified vs McMillan.dat)


def read_a2f(path):
    w, f = [], []
    for line in open(path):
        t = line.split()
        if len(t) >= 2 and not line.lstrip().startswith("#"):
            try:
                wv, fv = float(t[0]), float(t[1])
            except ValueError:
                continue
            w.append(wv)
            f.append(fv)
    w = np.asarray(w) * RY_TO_THZ
    f = np.asarray(f)
    m = w > CUT_K / THZ_TO_K
    return w[m], f[m]


def lam_wlog(w_thz, y):
    """Allen-Dynes moments from an alpha^2F curve (any grid), omega_log in K."""
    m = w_thz > 1e-9
    w, y = w_thz[m], y[m]
    lam = 2.0 * np.trapezoid(y / w, w)
    wlog = np.exp((2.0 / lam) * np.trapezoid(np.log(w) * y / w, w)) * THZ_TO_K
    return lam, wlog


def cmd_a2f(dry_run=False):
    dirs = glob.glob(os.path.join(EPH, "a2f_raw", "batch-*", "*_agm*"))
    graph_dir = os.path.join(EPH, "graphs_v45_eph")
    centers = (np.arange(PHONON_N_BINS) + 0.5) * (PHONON_W_MAX_THZ / PHONON_N_BINS)
    n_ok = n_nograph = 0
    err_l, err_w = [], []
    for d in sorted(dirs):
        agm = d.rsplit("_", 1)[-1]
        a2f_path = os.path.join(d, "a2F.dos6")
        gpath = os.path.join(graph_dir, agm + ".json")
        if not os.path.exists(a2f_path):
            continue
        if not os.path.exists(gpath):
            n_nograph += 1
            continue
        w, f = read_a2f(a2f_path)
        binned = bin_spectrum(w, f)
        # binning-fidelity check vs the author-integrated values
        mc = open(os.path.join(d, "McMillan.dat")).read()
        try:
            la_ref = float(re.search(r"lambda\s*=\s*([\d.Ee+-]+)", mc).group(1))
            wl_ref = float(re.search(r"wlog\[K\]\s*=\s*([\d.Ee+-]+)", mc).group(1))
            la_b, wl_b = lam_wlog(centers, binned.astype(np.float64))
            if la_ref > 0 and wl_ref > 0:
                err_l.append((la_b - la_ref) / la_ref)
                err_w.append((wl_b - wl_ref) / wl_ref)
        except AttributeError:
            pass
        if not dry_run:
            g = json.load(open(gpath))
            g["a2f"] = [round(float(x), 8) for x in binned]
            tmp = gpath + ".tmp"
            json.dump(g, open(tmp, "w"))
            os.replace(tmp, gpath)
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  baked {n_ok}", flush=True)
    err_l, err_w = np.array(err_l), np.array(err_w)
    print(f"a2f bake: {n_ok} baked, {n_nograph} without graphs, "
          f"{len(dirs) - n_ok - n_nograph} without a2F.dos6")
    if len(err_l):
        print(f"binned-vs-McMillan lambda err: median {np.median(np.abs(err_l))*100:.2f}% "
              f"p95 {np.percentile(np.abs(err_l),95)*100:.2f}% max {np.abs(err_l).max()*100:.2f}%")
        print(f"binned-vs-McMillan wlog   err: median {np.median(np.abs(err_w))*100:.2f}% "
              f"p95 {np.percentile(np.abs(err_w),95)*100:.2f}% max {np.abs(err_w).max()*100:.2f}%")


_FREQ_RE = re.compile(r"freq\s*\(\s*\d+\s*\)\s*=\s*(-?[\d.]+)\s*\[THz\]")
PHDOS_SIGMA_THZ = 0.15          # dense interpolated mesh -> tight smearing


def _phdos_one(d):
    """Worker: dense-mesh DOS for one material dir -> (agm, dos, kept, err) or
    (agm, None, msg, None) on failure."""
    sys.path.insert(0, os.path.join(_ROOT, "scripts"))
    from plot_phonon_dispersion import dense_phdos
    centers = (np.arange(PHONON_N_BINS) + 0.5) * (PHONON_W_MAX_THZ / PHONON_N_BINS)
    agm = d.rsplit("_", 1)[-1]
    try:
        dos, nat, kept, err = dense_phdos(d, centers, sigma=PHDOS_SIGMA_THZ)
        if not np.isfinite(err) or err > 0.02:
            return agm, None, f"grid validation err {err}", None
        return agm, dos, kept, err
    except Exception as e:  # noqa: BLE001 — per-material isolation
        return agm, None, str(e)[:120], None


def read_dyn_freqs(mat_dir):
    """(freqs_THz, weights, nat, grid_total) from a material's qe.dyn* files.
    One irreducible q per file; weight = its star multiplicity (count of
    'Dynamical  Matrix' blocks); frequencies parsed from the pre-diagonalized
    listing. grid_total from qe.dyn0 (n1*n2*n3) for the weight sanity check."""
    freqs, weights = [], []
    nat = grid_total = None
    d0 = os.path.join(mat_dir, "qe.dyn0")
    if os.path.exists(d0):
        with open(d0) as f:
            grid_total = int(np.prod([int(x) for x in f.readline().split()[:3]]))
    for p in sorted(glob.glob(os.path.join(mat_dir, "qe.dyn[1-9]*"))):
        txt = open(p).read()
        if nat is None:
            for line in txt.splitlines():
                t = line.split()
                if len(t) >= 3 and t[0].isdigit() and t[1].isdigit():
                    nat = int(t[1])
                    break
        mult = txt.count("Dynamical  Matrix in cartesian axes")
        fq = [float(x) for x in _FREQ_RE.findall(txt)]
        if mult and fq:
            freqs.append(np.asarray(fq))
            weights.append(mult)
    return freqs, np.asarray(weights, float), nat, grid_total


def cmd_phdos(dry_run=False):
    """Dense-mesh phonon DOS (interpolated force constants, basis recovered
    from the q set — lattice-convention-free) baked per material; each worker
    self-validates against QE's listed grid frequencies (skip if err > 0.02 THz)."""
    from multiprocessing import Pool
    dirs = sorted(d for d in glob.glob(os.path.join(EPH, "a2f_raw", "batch-*", "*_agm*"))
                  if glob.glob(os.path.join(d, "qe.dyn[1-9]*")))
    graph_dir = os.path.join(EPH, "graphs_v45_eph")
    n_ok = n_fail = 0
    kept_fr, errs = [], []
    with Pool(min(12, os.cpu_count() or 1)) as pool:
        for agm, dos, kept, err in pool.imap_unordered(_phdos_one, dirs, chunksize=8):
            gpath = os.path.join(graph_dir, agm + ".json")
            if dos is None or not os.path.exists(gpath):
                n_fail += 1
                if n_fail <= 15:
                    print(f"  skip {agm}: {kept}", flush=True)
                continue
            if not dry_run:
                g = json.load(open(gpath))
                g["ph_dos"] = [round(float(x), 8) for x in dos]
                tmp = gpath + ".tmp"
                json.dump(g, open(tmp, "w"))
                os.replace(tmp, gpath)
            kept_fr.append(kept)
            errs.append(err)
            n_ok += 1
            if n_ok % 500 == 0:
                print(f"  baked {n_ok}", flush=True)
    kept_fr, errs = np.array(kept_fr), np.array(errs)
    print(f"phdos bake (dense mesh): {n_ok} baked, {n_fail} skipped")
    if len(kept_fr):
        print(f"grid-validation err: median {np.median(errs):.5f} max {errs.max():.5f} THz")
        print(f"stable-mode fraction: median {np.median(kept_fr)*100:.2f}% "
              f"| <95% stable: {(kept_fr < 0.95).sum()} materials")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a2f", "phdos"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    (cmd_a2f if a.cmd == "a2f" else cmd_phdos)(dry_run=a.dry_run)
