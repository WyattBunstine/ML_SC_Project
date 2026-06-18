"""Fetch Materials Project total DOS and attach it to the relaxed MP graphs.

The DOS-bearing population for multitask pretraining is the RELAXED MP structures (the
MP-energy graphs), a separate masked-union member from the per-frame MPtrj targets. This
streams an MP index, fetches each material's DOS via the official MP-API, resamples the
total density onto a FIXED E_F-aligned grid (matching models/common/data.DOS_N_ENERGY),
and attaches it as graph["dos"] (atomic write, resumable). pack-dataset then carries it
into the DOS pack's meta.

    python main.py fetch-dos --index database/datafiles/MP_Energy/energies_MP_v4.pickle
    # then: python main.py pack-dataset --index <that index> --out <dos_pack_dir>

DOS coverage is a SUBSET of MP (not every material has an electronic-structure calc) and
only on the relaxed structure, so expect many `no_dos` skips — logged per-kind.

The energy grid + resampling (dos_to_grid / complete_dos_total) are pure functions, unit-
testable without the API. The fetch needs MP_API_KEY + network.
"""
import json
import os

import numpy as np

# Fixed E_F-aligned energy grid (eV relative to the Fermi level). N_ENERGY MUST match
# models/common/data.DOS_N_ENERGY (and the model's n_energy) for the target to line up.
DOS_EMIN, DOS_EMAX, N_ENERGY = -10.0, 5.0, 256
DOS_GRID = np.linspace(DOS_EMIN, DOS_EMAX, N_ENERGY)


def _get_api_key():
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise RuntimeError("Set MP_API_KEY (the Materials Project API key) to fetch DOS.")
    return key


def dos_to_grid(energies, total_density, efermi, broaden_ev=0.1):
    """Resample a total DOS onto DOS_GRID (E_F-aligned), with optional Gaussian broadening.

    energies/total_density: 1D arrays (raw MP grid); efermi: float. Returns a (N_ENERGY,)
    float32 density vector (clamped >= 0), or None when the DOS is DEGENERATE/unusable:
    fewer than 2 points (np.interp can't sample / a 1-point grid would fabricate a flat
    nonzero spectrum), a non-finite E_F, or no density inside [-10,+5] eV after alignment
    (an all-zero target would actively train the head toward zero). None -> the caller
    records a no_dos miss, so it isn't silently written as a covered-but-useless label."""
    e = np.asarray(energies, dtype=float)
    if e.size < 2 or efermi is None or not np.isfinite(efermi):
        return None
    e_rel = e - float(efermi)
    d = np.asarray(total_density, dtype=float)
    order = np.argsort(e_rel)
    g = np.interp(DOS_GRID, e_rel[order], d[order], left=0.0, right=0.0)
    if broaden_ev and broaden_ev > 0:
        de = (DOS_EMAX - DOS_EMIN) / (N_ENERGY - 1)
        sigma = max(broaden_ev / de, 1e-6)
        half = max(1, int(round(3 * sigma)))
        x = np.arange(-half, half + 1)
        kern = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kern /= kern.sum()
        # Edge-normalize: divide by the kernel mass that landed IN-grid so boundary bins
        # don't leak density into the implicit zero-padding (a ~13% edge underestimate).
        norm = np.convolve(np.ones_like(g), kern, mode="same")
        g = np.convolve(g, kern, mode="same") / np.clip(norm, 1e-12, None)
    g = np.clip(g, 0.0, None).astype(np.float32)
    if not g.sum() > 0:                          # no density in-window -> unusable
        return None
    return g


def complete_dos_total(cdos):
    """(energies, total density summed over spins, efermi) from a pymatgen CompleteDos.
    A None E_F is returned as NaN (dos_to_grid treats it as unusable) rather than raising."""
    energies = np.asarray(cdos.energies, dtype=float)
    total = np.zeros_like(energies)
    for spin_density in cdos.densities.values():
        total = total + np.asarray(spin_density, dtype=float)
    ef = cdos.efermi
    return energies, total, (float(ef) if ef is not None else float("nan"))


def _write_graph(graph, gp):
    tmp = gp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(graph, f)
    os.replace(tmp, gp)


def _looks_missing(exc):
    """A terminal 'this material has no DOS' API error vs a transient (retryable) one."""
    msg = str(exc).lower()
    return any(s in msg for s in ("404", "not found", "no electronic", "no dos", "no data"))


def fetch_and_attach_dos(index_path, broaden_ev=0.1, limit=None, retries=3):
    """Attach graph["dos"] (resampled total DOS) to each relaxed MP graph the index points
    at, by material id. Resumable on BOTH a written dos AND a no_dos sentinel
    (graph["dos_missing"]) — so a re-run doesn't re-query the ~majority of MP materials that
    have no DOS. Transient API errors are retried with backoff and left un-marked (retried
    next run); terminal-missing / degenerate DOS is recorded as no_dos. Per-kind tally."""
    import time
    import pandas as pd
    from mp_api.client import MPRester

    df = pd.read_pickle(index_path)
    counters = {k: 0 for k in ("streamed", "ok", "skip", "no_graph", "no_dos", "fail")}
    start = time.time()
    with MPRester(_get_api_key()) as mpr:
        for _, row in df.iterrows():
            if limit and counters["streamed"] >= limit:
                break
            counters["streamed"] += 1
            mid = str(row["id"]).replace(".cif", "")        # material id for the MP-API
            gp = row["graph_path"]
            if not os.path.exists(gp):
                counters["no_graph"] += 1
                continue
            with open(gp) as f:
                graph = json.load(f)
            if "dos" in graph or graph.get("dos_missing"):  # resumable: hit OR known miss
                counters["skip"] += 1
                continue
            grid, failed = None, False
            for attempt in range(retries):
                try:
                    cdos = mpr.get_dos_by_material_id(mid)
                    grid = (None if cdos is None
                            else dos_to_grid(*complete_dos_total(cdos), broaden_ev=broaden_ev))
                    break
                except Exception as exc:  # noqa: BLE001
                    if _looks_missing(exc):
                        break                               # terminal: no DOS -> grid stays None
                    if attempt == retries - 1:
                        failed = True
                        counters["fail"] += 1
                        print(f"  [fail] {mid}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                        break
                    time.sleep(2 ** attempt)                 # transient: backoff + retry
            if failed:
                continue                                    # un-marked -> retried next run
            if grid is None:                                # missing / degenerate -> sentinel
                graph["dos_missing"] = True
                _write_graph(graph, gp)
                counters["no_dos"] += 1
            else:
                graph["dos"] = grid.tolist()
                _write_graph(graph, gp)
                counters["ok"] += 1
            if counters["streamed"] % 500 == 0:
                rate = counters["streamed"] / max(time.time() - start, 1e-9)
                print(f"  {counters['streamed']} processed ({rate:.1f}/s)  {counters}", flush=True)
    print(f"DOS attach done: {counters}", flush=True)
    print(f"  coverage: {counters['ok']} graphs gained DOS; {counters['no_dos']} had none "
          f"(electronic-structure calc absent — a SUBSET of MP, as expected).")
    return counters
