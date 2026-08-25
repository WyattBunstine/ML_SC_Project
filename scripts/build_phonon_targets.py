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
PHDOS_SIGMA_THZ = 0.25          # smearing for the coarse (3^3-4^3) q grids


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
    dirs = glob.glob(os.path.join(EPH, "a2f_raw", "batch-*", "*_agm*"))
    graph_dir = os.path.join(EPH, "graphs_v45_eph")
    centers = (np.arange(PHONON_N_BINS) + 0.5) * (PHONON_W_MAX_THZ / PHONON_N_BINS)
    dw = PHONON_W_MAX_THZ / PHONON_N_BINS
    n_ok = n_nodyn = n_badgrid = 0
    imag_frac, clip_frac = [], []
    for d in sorted(dirs):
        agm = d.rsplit("_", 1)[-1]
        gpath = os.path.join(graph_dir, agm + ".json")
        freqs, w, nat, grid_total = read_dyn_freqs(d)
        if not freqs or nat is None or not os.path.exists(gpath):
            n_nodyn += 1
            continue
        if grid_total is not None and int(w.sum()) != grid_total:
            n_badgrid += 1                       # count, but keep — weights still relative
        allf = np.concatenate(freqs)
        allw = np.concatenate([np.full(len(f), ww) for f, ww in zip(freqs, w)])
        pos = allf > 0
        imag_frac.append(1.0 - allw[pos].sum() / allw.sum())
        # Gaussian smear onto the fixed grid; renormalize the KEPT weight to
        # 3*nat*(kept fraction) so dropped imaginary modes don't inflate the rest.
        dos = np.zeros(PHONON_N_BINS)
        for f0, ww in zip(allf[pos], allw[pos]):
            dos += ww * np.exp(-0.5 * ((centers - f0) / PHDOS_SIGMA_THZ) ** 2)
        dos /= (PHDOS_SIGMA_THZ * np.sqrt(2 * np.pi))
        clip_frac.append(float(allw[pos & (allf > PHONON_W_MAX_THZ)].sum() / allw.sum()))
        target_modes = 3.0 * nat * (allw[pos].sum() / allw.sum())
        integ = dos.sum() * dw
        if integ > 0:
            dos *= target_modes / integ
        if not dry_run:
            g = json.load(open(gpath))
            g["ph_dos"] = [round(float(x), 8) for x in dos]
            tmp = gpath + ".tmp"
            json.dump(g, open(tmp, "w"))
            os.replace(tmp, gpath)
        n_ok += 1
        if n_ok % 500 == 0:
            print(f"  baked {n_ok}", flush=True)
    imag_frac = np.array(imag_frac)
    print(f"phdos bake: {n_ok} baked, {n_nodyn} without dyn/graph, "
          f"{n_badgrid} weight-sum != dyn0 grid")
    if len(imag_frac):
        print(f"imaginary-mode weight: median {np.median(imag_frac)*100:.2f}% "
              f"p95 {np.percentile(imag_frac,95)*100:.2f}% "
              f"| >5% imag: {(imag_frac>0.05).sum()} materials "
              f"| clipped >60THz: {(np.array(clip_frac)>0.01).sum()} materials")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a2f", "phdos"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    (cmd_a2f if a.cmd == "a2f" else cmd_phdos)(dry_run=a.dry_run)
