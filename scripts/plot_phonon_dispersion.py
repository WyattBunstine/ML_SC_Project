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
        at = None         # non-cubic: caller must use recover_basis (q-set derived)
    masses, syms = {}, {}
    for k in range(ntyp):
        t = lines[idx + k].split("'")
        masses[int(t[0])] = float(t[2])
        syms[int(t[0])] = t[1].strip()
    idx += ntyp
    tau, types = [], []
    for k in range(nat):
        t = lines[idx + k].split()
        types.append(int(t[1]))
        tau.append([float(x) for x in t[2:5]])
    tau = np.array(tau)
    m = np.array([masses[t] for t in types])
    symbols = [syms[t] for t in types]

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
    return at, tau, m, qd, freq_ref, nat, symbols


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
                        # minimal-image fold of R + tau_j - tau_i; no folding
                        # freedom along unsampled (N_i = 1) directions
                        best, cands = None, []
                        for m1 in ((-1, 0, 1) if N[0] > 1 else (0,)):
                            for m2 in ((-1, 0, 1) if N[1] > 1 else (0,)):
                                for m3 in ((-1, 0, 1) if N[2] > 1 else (0,)):
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


def recover_basis(qs, tol=1e-5):
    """Recover the q-grid's generating basis PURELY from the q-point set —
    lattice-convention-free (works for every ibrav, unlike hand-wired vectors).

    The full-grid q's form a lattice L generated by g_i = b_i/N_i. Greedily pick
    the 3 shortest independent candidates (q's + pairwise differences) under the
    constraint that EVERY q is an integer combination; returns (G rows g_i,
    N per generator, A rows a_j in alat units with g_i . a_j = delta_ij/N_i).
    Self-validating downstream: grid-point frequencies must reproduce QE's."""
    qs = np.asarray(qs, float)
    qs = qs[np.linalg.norm(qs, axis=1) > tol]
    cands = [qs]
    for k in range(min(len(qs), 60)):
        cands.append(qs - qs[k])
    cands = np.unique(np.round(np.concatenate(cands), 7), axis=0)
    cands = cands[np.linalg.norm(cands, axis=1) > tol]
    cands = cands[np.argsort(np.linalg.norm(cands, axis=1))]
    basis = []
    for c in cands:
        trial = basis + [c]
        if np.linalg.matrix_rank(np.array(trial), tol=tol) != len(trial):
            continue
        basis = trial
        if len(basis) == 3:
            G = np.array(basis)
            M = qs @ np.linalg.inv(G)
            if np.abs(M - np.round(M)).max() < 1e-4:
                break
            basis = basis[:2]     # last pick fails integrality: try the next
    if len(basis) < 3:
        # Rank-deficient q set: grids with N_i = 1 (e.g. 5x5x1 layered
        # sampling) span only a plane/line. The unsampled directions carry no
        # phase information (their R components are always 0), so completing
        # the basis with null-space unit vectors is exact, not approximate.
        if basis:
            _, _, Vt = np.linalg.svd(np.array(basis).reshape(len(basis), 3))
            basis = list(np.array(basis)) + list(Vt[len(basis):])
        else:
            basis = list(np.eye(3))
    G = np.array(basis)
    return G


def assign_grid(G, qs, grid_dims):
    """Assign dyn0's (N1,N2,N3) to the recovered generators: the correct
    permutation makes the coefficient residues cover the full N1xN2xN3 grid."""
    from itertools import permutations
    M = np.round(np.concatenate([qs, [[0, 0, 0]]]) @ np.linalg.inv(G)).astype(int)
    M = np.unique(M, axis=0)
    for perm in permutations(grid_dims):
        N = np.array(perm, int)
        if len(np.unique(np.mod(M, N), axis=0)) == int(np.prod(N)) and \
           len(M) <= int(np.prod(N)):
            B = G * N[:, None]             # full reciprocal vectors b_i = N_i g_i
            A = np.linalg.inv(B).T         # a_j rows: b_i . a_j = delta_ij
            return N, A
    raise ValueError(f"no grid-dim assignment of {grid_dims} covers the q set")


def dense_phdos(mat_dir, centers, sigma=0.15, mesh_cap=16, per_atom=False,
                site_proj=False, site_centers=None):
    """Phonon DOS on the fixed grid from a DENSE interpolated q-mesh (vectorized
    D(q) assembly + batched eigh). Returns (dos, nat, kept_frac, grid_err) or
    raises. grid_err = max |interpolated - QE-listed| at the calculated q's
    (pre-ASR would be exact; ASR redistribution keeps this ~0.3 THz).

    site_proj: additionally return the SITE-projected DOS reduced per ELEMENT
    ({symbol: (len(site_centers),) fp32}) — mode weights are the squared
    eigenvector components of each atom (orthonormal in mass-weighted
    coordinates, so every mode partitions exactly). Per-ATOM normalization
    (each atom integrates to ~3*kept states) so no cell atom count enters —
    sidesteps the primitive-vs-conventional QE/graph cell trap. Per-element
    reduction is cell-convention- and orientation-proof; intra-element site
    distinction is deliberately out of scope (v2 = exact site matching)."""
    at, tau, m, qd, freq_ref, nat, symbols = parse_dyn_dir(mat_dir)
    qs = np.array(list(qd.keys()))
    grid_dims = [int(x) for x in
                 open(os.path.join(mat_dir, "qe.dyn0")).readline().split()[:3]]
    G = recover_basis(qs)
    N, A = assign_grid(G, qs, grid_dims)
    if int(np.prod(N)) != len(qs):
        raise ValueError(f"grid {N} inconsistent with {len(qs)} q-points")
    entries, asr = force_constants(A, tau, qd, nat, N)
    # per-material validation at the calculated q's (no ASR -> must be exact)
    zero = [a * 0 for a in asr]
    errs = [np.abs(np.sort(freqs_at(np.array(q), entries, zero, m, nat))
                   - np.sort(np.array(r))).max() for q, r in freq_ref.items()]
    grid_err = max(errs) if errs else float("nan")
    # dense mesh (offset to avoid duplicating the calculated points exactly);
    # unsampled (N_i = 1) directions get a single point — interpolation is
    # constant along them by construction
    Mm = np.where(N == 1, 1, np.minimum(mesh_cap, np.maximum(12, 3 * N)))
    fr = [(np.arange(Mi) + 0.5) / Mi for Mi in Mm]
    mesh = np.array(np.meshgrid(*fr, indexing="ij")).reshape(3, -1).T @ (G * N[:, None])
    # group entries by (i, j) pair for vectorized assembly
    from collections import defaultdict
    groups = defaultdict(lambda: [[], [], []])
    for i, j, Rc, w, blk in entries:
        g = groups[(i, j)]
        g[0].append(Rc); g[1].append(w); g[2].append(blk)
    nq = len(mesh)
    D = np.zeros((nq, 3 * nat, 3 * nat), complex)
    for (i, j), (Rcs, ws, blks) in groups.items():
        Rcs = np.array(Rcs); ws = np.array(ws); blks = np.array(blks)
        ph = np.exp(2j * np.pi * (mesh @ Rcs.T)) * ws          # (nq, k)
        D[:, 3 * i:3 * i + 3, 3 * j:3 * j + 3] += np.tensordot(ph, blks, axes=([1], [0]))
    for i in range(nat):
        D[:, 3 * i:3 * i + 3, 3 * i:3 * i + 3] -= asr[i]
    Msq = np.sqrt(np.outer(np.repeat(m, 3), np.repeat(m, 3)))
    Dh = (D + np.conj(np.transpose(D, (0, 2, 1)))) / 2 / Msq
    if site_proj:
        ev, Vec = np.linalg.eigh(Dh)
        # per-atom mode weights: rows of V grouped by atom, summed over the 3
        # cartesian components -> (nq, nat, modes); columns are orthonormal so
        # weights partition each mode exactly (sum over atoms == 1)
        w_at = (np.abs(Vec) ** 2).reshape(nq, nat, 3, 3 * nat).sum(axis=2)
    else:
        ev = np.linalg.eigvalsh(Dh)
    freqs = (np.sign(ev) * np.sqrt(np.abs(ev)) * RY_TO_THZ).ravel()
    pos = freqs > 0
    kept = pos.mean()
    dos = np.zeros(len(centers))
    # chunked gaussian accumulation (freqs can be ~200k)
    for c0 in range(0, pos.sum(), 20000):
        f = freqs[pos][c0:c0 + 20000]
        dos += np.exp(-0.5 * ((centers[:, None] - f[None, :]) / sigma) ** 2).sum(axis=1)
    dos /= sigma * np.sqrt(2 * np.pi)
    dw = centers[1] - centers[0]
    # per_atom: integral = 3*kept states PER ATOM — the consumer must rescale by
    # ITS OWN cell's atom count. The QE/phonopy cell and the graph cell are NOT
    # always the same (primitive vs conventional: ~5% of the Cerqueira corpus,
    # e.g. agm002137061 nat 3 vs graph 6 — review 2026-08-26), so normalizing
    # to this cell's 3*nat and dividing by the graph's n_atoms downstream was
    # silently wrong by an integer factor for those materials.
    target = 3.0 * kept * (1.0 if per_atom else nat)
    if dos.sum() * dw > 0:
        dos *= target / (dos.sum() * dw)
    if not site_proj:
        return dos.astype(np.float32), nat, kept, grid_err
    sc = np.asarray(site_centers if site_centers is not None else centers, float)
    fpos = freqs[pos]
    wpos = w_at.transpose(1, 0, 2).reshape(nat, -1)[:, pos]      # (nat, n_pos_modes)
    spec = np.zeros((nat, len(sc)))
    for c0 in range(0, fpos.size, 20000):
        f = fpos[c0:c0 + 20000]
        g = np.exp(-0.5 * ((sc[:, None] - f[None, :]) / sigma) ** 2)
        spec += wpos[:, c0:c0 + 20000] @ g.T
    spec /= sigma * np.sqrt(2 * np.pi)
    dws = sc[1] - sc[0]
    tot = spec.sum() * dws
    if tot > 0:
        spec *= (3.0 * kept * nat) / tot        # per-atom convention: each ~3*kept
    site = {}
    for el in sorted(set(symbols)):
        rows = [k for k, s in enumerate(symbols) if s == el]
        site[el] = spec[rows].mean(axis=0).astype(np.float32)
    return dos.astype(np.float32), nat, kept, grid_err, site


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
    at, tau, m, qd, freq_ref, nat, symbols = parse_dyn_dir(mat_dir)
    grid = [int(x) for x in open(os.path.join(mat_dir, "qe.dyn0")).readline().split()[:3]]
    print(f"{os.path.basename(mat_dir)}: nat {nat}, grid {grid}, {len(qd)} q-points")
    if at is None:
        # non-cubic ibrav: same q-set-derived basis the bake path uses
        qs = np.array(list(qd.keys()))
        grid, at = assign_grid(recover_basis(qs), qs, grid)
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
