"""Phonon dispersions from Cerqueira qe.dyn* files (q2r->matdyn in miniature).

The dyn files carry D(q) for EVERY q in each star, i.e. the complete grid:
inverse-FT to real-space force constants (minimal-image folded, simple acoustic
sum rule), FT back to arbitrary q along a high-symmetry path. Self-validating:
diagonalization AT the grid q-points must reproduce the frequencies QE printed
in the same files.

  python scripts/plot_phonon_dispersion.py <material_dir> [--path GXMGR|GHNGPH]
"""
import glob
import os
import re
import sys

import numpy as np

RY_TO_THZ = 13605.693122994 / 4.135667696


def parse_dyn_dir(mat_dir):
    """(lattice(alat units,rows), tau(alat cart), masses, {q: D}) from qe.dyn*."""
    first = sorted(glob.glob(os.path.join(mat_dir, "qe.dyn[1-9]*")))[0]
    lines = open(first).read().splitlines()
    ntyp, nat, ibrav = (int(x) for x in lines[2].split()[:3])
    celldm = [float(x) for x in lines[2].split()[3:9]]
    at = None
    idx = 3
    if ibrav == 0:
        # "Basis vectors" + 3 rows in alat units
        idx += 1
        at = np.array([[float(x) for x in lines[idx + k].split()] for k in range(3)])
        idx += 3
    elif ibrav == 1:
        at = np.eye(3)
    elif ibrav == 2:      # fcc (QE convention)
        at = 0.5 * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], float)
    elif ibrav == 3:      # bcc (QE convention)
        at = 0.5 * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]], float)
    else:
        raise ValueError(f"ibrav {ibrav} not wired (add its lattice vectors)")
    masses = {}
    for k in range(ntyp):
        t = lines[idx + k].split("'")
        masses[int(t[0])] = float(t[2])
    idx += ntyp
    tau, types = [], []
    for k in range(nat):
        t = lines[idx + k].split()
        types.append(int(t[1]))
        tau.append([float(x) for x in t[2:5]])
    tau = np.array(tau)
    m = np.array([masses[t] for t in types])

    qd = {}
    freq_ref = {}
    for p in sorted(glob.glob(os.path.join(mat_dir, "qe.dyn[1-9]*"))):
        txt = open(p).read()
        for block in txt.split("Dynamical  Matrix in cartesian axes")[1:]:
            head = block.split("Diagonalizing")[0]
            qm = re.search(r"q = \(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", head)
            q = tuple(round(float(x), 9) for x in qm.groups())
            nums = re.findall(r"-?\d+\.\d+(?:[Ee][+-]?\d+)?", head[qm.end():])
            vals = np.array([float(x) for x in nums])
            # per atom pair: 'i j' ints then 9 complex = 18 floats; the ints are
            # not matched by the float regex, so vals is exactly nat^2*18 long
            D = np.zeros((3 * nat, 3 * nat), complex)
            k = 0
            for i in range(nat):
                for j in range(nat):
                    b = vals[k:k + 18].reshape(3, 3, 2)
                    D[3 * i:3 * i + 3, 3 * j:3 * j + 3] = b[..., 0] + 1j * b[..., 1]
                    k += 18
            qd[q] = D
        fq = re.search(r"Diagonalizing.*", txt, re.S)
        if fq:
            qm = re.search(r"q = \(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)", fq.group(0))
            q = tuple(round(float(x), 9) for x in qm.groups())
            freq_ref[q] = [float(x) for x in
                           re.findall(r"freq \(\s*\d+\) =\s*(-?[\d.]+) \[THz\]", fq.group(0))]
    return at, tau, m, qd, freq_ref, nat


def force_constants(at, tau, qd, nat, grid):
    """Minimal-image-folded real-space force constants: list of
    (i, j, R_cart, weight, C 3x3) with simple ASR applied."""
    qs = np.array(list(qd.keys()))
    Ds = np.array([qd[tuple(q)] for q in qs])
    N = np.array(grid)
    # C(R_frac) on the grid (plain inverse DFT; exact at grid q by construction)
    entries = []
    Csum = np.zeros((nat, 3, 3))
    for n1 in range(N[0]):
        for n2 in range(N[1]):
            for n3 in range(N[2]):
                Rf = np.array([n1, n2, n3], float)
                Rc0 = Rf @ at
                ph = np.exp(-2j * np.pi * (qs @ Rc0))
                C = (Ds * ph[:, None, None]).mean(axis=0)
                for i in range(nat):
                    for j in range(nat):
                        blk = C[3 * i:3 * i + 3, 3 * j:3 * j + 3].real
                        Csum[i] += blk
                        # minimal-image fold of R + tau_j - tau_i
                        best, cands = None, []
                        for m1 in (-1, 0, 1):
                            for m2 in (-1, 0, 1):
                                for m3 in (-1, 0, 1):
                                    Rc = (Rf + N * np.array([m1, m2, m3])) @ at
                                    d = np.linalg.norm(Rc + tau[j] - tau[i])
                                    if best is None or d < best - 1e-6:
                                        best, cands = d, [Rc]
                                    elif abs(d - best) <= 1e-6:
                                        cands.append(Rc)
                        for Rc in cands:
                            entries.append((i, j, Rc, 1.0 / len(cands), blk))
    # simple ASR: the self block absorbs minus the row sum
    asr = [np.zeros((3, 3)) for _ in range(nat)]
    for i, j, Rc, w, blk in entries:
        asr[i] += w * blk
    return entries, asr


def freqs_at(q, entries, asr, masses, nat):
    D = np.zeros((3 * nat, 3 * nat), complex)
    for i, j, Rc, w, blk in entries:
        D[3 * i:3 * i + 3, 3 * j:3 * j + 3] += w * blk * np.exp(2j * np.pi * np.dot(q, Rc))
    for i in range(nat):
        D[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= asr[i]
    Msq = np.sqrt(np.outer(np.repeat(masses, 3), np.repeat(masses, 3)))
    ev = np.linalg.eigvalsh((D + D.conj().T) / 2 / Msq)
    return np.sign(ev) * np.sqrt(np.abs(ev)) * RY_TO_THZ


PATHS = {
    "GXMGR": [("Γ", (0, 0, 0)), ("X", (0, .5, 0)), ("M", (.5, .5, 0)),
              ("Γ", (0, 0, 0)), ("R", (.5, .5, .5))],
    "GHNGPH": [("Γ", (0, 0, 0)), ("H", (0, 0, 1)), ("N", (0, .5, .5)),
               ("Γ", (0, 0, 0)), ("P", (.5, .5, .5)), ("H", (0, 0, 1))],
}


def band_path(points, entries, asr, masses, nat, npts=60):
    xs, ws, ticks = [], [], [0.0]
    x = 0.0
    for (la, qa), (lb, qb) in zip(points[:-1], points[1:]):
        qa, qb = np.array(qa, float), np.array(qb, float)
        seg = np.linalg.norm(qb - qa)
        for t in np.linspace(0, 1, npts, endpoint=False):
            q = qa + t * (qb - qa)
            xs.append(x + t * seg)
            ws.append(freqs_at(q, entries, asr, masses, nat))
        x += seg
        ticks.append(x)
    xs.append(x)
    ws.append(freqs_at(np.array(points[-1][1], float), entries, asr, masses, nat))
    return np.array(xs), np.array(ws), ticks, [p[0] for p in points]


def run(mat_dir, path_key):
    at, tau, m, qd, freq_ref, nat = parse_dyn_dir(mat_dir)
    grid = [int(x) for x in open(os.path.join(mat_dir, "qe.dyn0")).readline().split()[:3]]
    print(f"{os.path.basename(mat_dir)}: nat {nat}, grid {grid}, {len(qd)} q-points")
    entries, asr = force_constants(at, tau, qd, nat, grid)
    errs = []
    for q, ref in freq_ref.items():
        mine = freqs_at(np.array(q), entries, asr, m, nat)
        errs.append(np.abs(np.sort(mine) - np.sort(np.array(ref))).max())
    print(f"grid-point validation vs QE-listed freqs: max |err| {max(errs):.4f} THz "
          f"over {len(errs)} irreducible q")
    return band_path(PATHS[path_key], entries, asr, m, nat)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "GXMGR")
