"""Bake phonon-spectrum targets into graph JSONs (tasks #7/#8/#9).

Subcommands:
  a2f    — Cerqueira alpha^2F: parse a2F.dos6 (published-value smearing) from
           EPH_Cerqueira/a2f_raw/batch-*/<Formula>_<agm>/, apply the verified
           12 K low-omega cutoff, bin onto the shared PHONON grid
           (data.bin_spectrum), write g["a2f"] into graphs_v45_eph/<agm>.json.
           VALIDATES corpus-wide: lambda/omega_log recomputed from the BINNED
           spectrum vs McMillan.dat — the binning-fidelity report.
           Re-runnable (idempotent: overwrites the key).

  python scripts/build_phonon_targets.py a2f [--dry-run]
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["a2f"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cmd_a2f(dry_run=a.dry_run)
