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
    float32 vector, clamped >= 0 (a density). Outside the source range -> 0."""
    e_rel = np.asarray(energies, dtype=float) - float(efermi)
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
        g = np.convolve(g, kern, mode="same")
    return np.clip(g, 0.0, None).astype(np.float32)


def complete_dos_total(cdos):
    """(energies, total density summed over spins, efermi) from a pymatgen CompleteDos."""
    energies = np.asarray(cdos.energies, dtype=float)
    total = np.zeros_like(energies)
    for spin_density in cdos.densities.values():
        total = total + np.asarray(spin_density, dtype=float)
    return energies, total, float(cdos.efermi)


def fetch_and_attach_dos(index_path, broaden_ev=0.1, limit=None):
    """Attach graph["dos"] (resampled total DOS) to each relaxed MP graph the index points
    at, by material id. Resumable (skips graphs already carrying dos); per-kind tally."""
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
            try:
                with open(gp) as f:
                    graph = json.load(f)
                if "dos" in graph:                          # resumable
                    counters["skip"] += 1
                    continue
                cdos = mpr.get_dos_by_material_id(mid)
                if cdos is None:
                    counters["no_dos"] += 1
                    continue
                en, total, ef = complete_dos_total(cdos)
                graph["dos"] = dos_to_grid(en, total, ef, broaden_ev).tolist()
                tmp = gp + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(graph, f)
                os.replace(tmp, gp)
                counters["ok"] += 1
            except Exception as exc:  # noqa: BLE001 — DOS missing/odd is common; keep going
                msg = str(exc)
                if "404" in msg or "not found" in msg.lower() or "No electronic" in msg:
                    counters["no_dos"] += 1
                else:
                    counters["fail"] += 1
                    print(f"  [fail] {mid}: {type(exc).__name__}: {msg[:120]}", flush=True)
            if counters["streamed"] % 500 == 0:
                rate = counters["streamed"] / max(time.time() - start, 1e-9)
                print(f"  {counters['streamed']} processed ({rate:.1f}/s)  {counters}", flush=True)
    print(f"DOS attach done: {counters}", flush=True)
    print(f"  coverage: {counters['ok']} graphs gained DOS; {counters['no_dos']} had none "
          f"(electronic-structure calc absent — a SUBSET of MP, as expected).")
    return counters
