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


def _is_validation_error(exc):
    """A pydantic ValidationError from the MP client's document model — deterministic
    (the server response doesn't match emmet-core's schema), so retrying never helps."""
    return "validationerror" in type(exc).__name__.lower()


def _dos_object(mpr, mid):
    """CompleteDos for one material, downloaded from MP's AWS open-data store, which keys
    DOS objects by MATERIAL ID — `dos/<mid>.json.gz` (verified by listing the bucket). The
    stock get_dos_by_material_id instead resolves a task_id and fetches dos/<task_id>.json.gz,
    which 404s for everything (wrong key scheme + an AlphaID->numeric mangling on top). We go
    straight to the material-id key. Returns a pymatgen CompleteDos, or None on a 404 (no
    object for this material). The stored object is either the CompleteDos directly or wrapped
    as {"data": dos, ...} — handle both."""
    from mp_api.client.core.utils import load_json
    dr = mpr.materials.electronic_structure_dos
    # Let a 404 ("No object found") propagate: these materials passed the has-DOS prefilter,
    # so an absent object means "DOS calc exists but its object isn't in open-data" -> the
    # caller records it as no_object (distinct from no DOS calc at all).
    res = dr._query_open_data(
        bucket="materialsproject-parsed",
        key=f"dos/{mid}.json.gz",
        decoder=lambda x: load_json(x, deser=True))
    obj = res[0][0] if (res and res[0]) else None
    if isinstance(obj, dict) and "data" in obj:           # unwrap {"data": dos, ...}
        obj = obj["data"]
    return obj


def diagnose_dos(api_key, mid):
    """Probe the open-data DOS coverage: test the requested material plus canonical materials
    (Silicon, diamond, GaAs, MgO) and a known-present high-id key, then report the bucket's id
    range — to tell whether the dos/<mid>.json.gz mirror is missing common materials or only we
    are:  python main.py fetch-dos --diagnose mp-11944"""
    from mp_api.client import MPRester
    # Canonical, heavily-viewed DOS materials — if THESE 404, the open-data mirror is the
    # bottleneck, not our dataset. mp-1000000 is a key the listing proved is present.
    canon = ["mp-149", "mp-66", "mp-2534", "mp-1265", "mp-1000000"]
    with MPRester(api_key, use_document_model=False) as mpr:
        dr = mpr.materials.electronic_structure_dos
        print("  DOS object presence (dos/<mid>.json.gz):")
        for test_mid in dict.fromkeys([mid] + canon):       # requested first, dedup, keep order
            try:
                cdos = _dos_object(mpr, test_mid)
                grid = None if cdos is None else dos_to_grid(*complete_dos_total(cdos))
                print(f"    {test_mid:>12}: OK ({type(cdos).__name__}, "
                      f"grid {'present' if grid is not None else 'degenerate'})")
            except Exception as exc:  # noqa: BLE001
                miss = "no object found" in str(exc).lower()
                print(f"    {test_mid:>12}: {'404 absent' if miss else 'FAIL ' + type(exc).__name__ + ': ' + str(exc)[:70]}")
        # What id range does the bucket's dos/ prefix actually cover? (lexicographic first key)
        try:
            r = dr.s3_client.list_objects_v2(Bucket="materialsproject-parsed",
                                             Prefix="dos/mp-", MaxKeys=3)
            print(f"  bucket first dos/ keys (lexicographic): "
                  f"{[o['Key'] for o in r.get('Contents', [])]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  bucket list failed: {type(exc).__name__}: {str(exc)[:90]}")
        print("  (canonical materials OK -> scheme works, our experimental set is just poorly "
              "mirrored; canonical materials 404 -> the open-data DOS mirror itself is incomplete.)")


def _has_dos_props(doc):
    """True if a summary doc's has_props lists 'dos'. Tolerates the field being a list
    of enums/strings or a bool-valued dict across mp_api/emmet versions."""
    hp = getattr(doc, "has_props", None)
    if hp is None:
        return False
    if isinstance(hp, dict):
        return bool(hp.get("dos"))
    try:
        return any(str(getattr(x, "value", x)) == "dos" for x in hp)
    except TypeError:
        return False


def _materials_with_dos(api_key, mids, chunk=1000):
    """The subset of `mids` whose MP record actually HAS a computed DOS — found with bulk
    summary queries (one call per ~1000 ids) instead of a full-DOS download per material.
    DOS exists for a SUBSET of MP, so this lets the no-DOS majority be marked `dos_missing`
    without ever downloading a (large) DOS object. Tries the server-side has_props filter
    first; falls back to fetching has_props and filtering locally. Raises on total failure
    so the caller can degrade to fetching everything in parallel."""
    from mp_api.client import MPRester

    have = set()
    with MPRester(api_key) as mpr:
        for i in range(0, len(mids), chunk):
            ch = mids[i:i + chunk]
            try:                                            # server-side filter (cheapest)
                docs = mpr.materials.summary.search(
                    material_ids=ch, has_props=["dos"], fields=["material_id"])
                have.update(str(d.material_id) for d in docs)
            except Exception:                               # noqa: BLE001 — older client / enum
                docs = mpr.materials.summary.search(
                    material_ids=ch, fields=["material_id", "has_props"])
                have.update(str(d.material_id) for d in docs if _has_dos_props(d))
    return have


def _fetch_dos_grid(client_factory, mid, broaden_ev, retries):
    """Fetch + resample ONE material's total DOS. Returns (grid, outcome, exc):
      grid set, outcome None          -> success
      None, "no_dos",   None          -> no DOS doc / no task_id / degenerate spectrum
      None, "no_object", exc          -> DOS calc exists but its object is purged from MP's
                                         store ('No object found' for every candidate id —
                                         common for old calcs); deterministic, settle + flag
      None, "fail",     exc           -> transient (retries exhausted) or unexpected; un-marked
    Pure of file I/O so it is safe in a thread pool — the caller serializes the writes."""
    import time
    for attempt in range(retries):
        try:
            cdos = _dos_object(client_factory(), mid)
            if cdos is None:
                return None, "no_dos", None
            grid = dos_to_grid(*complete_dos_total(cdos), broaden_ev=broaden_ev)
            return (grid, None, None) if grid is not None else (None, "no_dos", None)
        except Exception as exc:  # noqa: BLE001
            if _looks_missing(exc):
                return None, "no_dos", None                 # genuinely no DOS
            if "no object found" in str(exc).lower():
                return None, "no_object", exc               # object purged -> settle + flag
            if _is_validation_error(exc) or attempt == retries - 1:
                return None, "fail", exc                    # un-marked, retried next run
            time.sleep(2 ** attempt)                        # transient: backoff + retry


def fetch_and_attach_dos(index_path, broaden_ev=0.1, limit=None, retries=3, workers=8,
                         retry_no_object=False):
    """Attach graph["dos"] (resampled total DOS) to each relaxed MP graph the index points
    at, by material id. Resumable on BOTH a written dos AND a no_dos sentinel
    (graph["dos_missing"]) — so a re-run doesn't re-query materials already settled.

    Two speedups over a naive per-material loop: (1) a BULK pre-filter (`_materials_with_dos`)
    settles the no-DOS majority from cheap metadata queries, with NO full-DOS download; (2)
    the remaining real DOS fetches run across a thread pool (`workers`) — the work is
    network-latency-bound, so concurrency is a near-linear win. Writes stay serialized on the
    main thread (atomic tmp+replace). Transient API errors retry with backoff and are left
    un-marked; terminal-missing / degenerate DOS is recorded as no_dos. Per-kind tally."""
    import threading
    import time
    import concurrent.futures as cf
    import pandas as pd
    from mp_api.client import MPRester

    df = pd.read_pickle(index_path)
    counters = {k: 0 for k in ("streamed", "ok", "skip", "no_graph",
                               "no_dos", "no_object", "fail")}
    start = time.time()

    # 1) Worklist: the rows that still need a fetch (resumable skips + missing graphs).
    work = []                                               # [(mid, graph_path), ...]
    for _, row in df.iterrows():
        if limit and counters["streamed"] >= limit:
            break
        counters["streamed"] += 1
        gp = row["graph_path"]
        if not os.path.exists(gp):
            counters["no_graph"] += 1
            continue
        with open(gp) as f:
            graph = json.load(f)
        if "dos" in graph:                                  # already attached -> skip
            counters["skip"] += 1
            continue
        if graph.get("dos_missing"):
            # `--retry-no-object` clears the no_object sentinels a PRIOR (wrong-key-scheme)
            # run wrote, so the corrected material-id fetch re-attempts them. Genuine no_dos
            # (no DOS calc per has_props) stays settled.
            if retry_no_object and graph.get("dos_skip_reason") == "no_object":
                graph.pop("dos_missing", None)
                graph.pop("dos_skip_reason", None)
                _write_graph(graph, gp)
            else:
                counters["skip"] += 1
                continue
        work.append((str(row["id"]).replace(".cif", ""), gp))

    api_key = _get_api_key()

    # 2) Bulk pre-filter: which of these materials actually have a DOS? Mark the rest
    #    dos_missing with no DOS download. Best-effort — degrade to fetching all on failure.
    to_fetch = work
    if work:
        try:
            have = _materials_with_dos(api_key, [m for m, _ in work])
            to_fetch, no_dos_pre = [], []
            for mid, gp in work:
                (to_fetch if mid in have else no_dos_pre).append((mid, gp))
            for _mid, gp in no_dos_pre:                     # settle misses, no API call
                with open(gp) as f:
                    graph = json.load(f)
                graph["dos_missing"] = True
                _write_graph(graph, gp)
                counters["no_dos"] += 1
            print(f"  pre-filter: {len(to_fetch)} have DOS, {len(no_dos_pre)} have none "
                  f"(of {len(work)} to settle)", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] DOS pre-filter unavailable ({type(exc).__name__}: "
                  f"{str(exc)[:100]}); fetching all {len(work)} in parallel", flush=True)
            to_fetch = work

    # 3) Parallel DOS fetch (network-bound). Each worker thread gets its own MPRester;
    #    results are consumed in order on this thread, which owns all the file writes.
    tls = threading.local()

    def client_factory():
        c = getattr(tls, "mpr", None)
        if c is None:
            # RAW mode: the DOS-summary query must skip the document-model validation
            # that the emmet-core/server task_id mismatch trips over (see _dos_object).
            c = tls.mpr = MPRester(api_key, use_document_model=False)
        return c

    def task(item):
        mid, gp = item
        grid, outcome, exc = _fetch_dos_grid(client_factory, mid, broaden_ev, retries)
        return gp, mid, grid, outcome, exc

    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for gp, mid, grid, outcome, exc in ex.map(task, to_fetch):
            done += 1
            if outcome == "fail":                           # transient/unexpected -> un-marked
                counters["fail"] += 1
                print(f"  [fail] {mid}: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
                continue
            with open(gp) as f:
                graph = json.load(f)
            if grid is not None:
                graph["dos"] = grid.tolist()
                counters["ok"] += 1
            else:                                           # settle (resumable skip) + flag why
                graph["dos_missing"] = True
                if outcome == "no_object":                  # DOS calc exists but object purged
                    graph["dos_skip_reason"] = "no_object"
                    counters["no_object"] += 1
                else:
                    counters["no_dos"] += 1
            _write_graph(graph, gp)
            if done % 500 == 0:
                rate = done / max(time.time() - start, 1e-9)
                print(f"  {done}/{len(to_fetch)} fetched ({rate:.1f}/s)  {counters}", flush=True)

    print(f"DOS attach done in {time.time() - start:.0f}s: {counters}", flush=True)
    print(f"  coverage: {counters['ok']} graphs gained DOS; {counters['no_dos']} have no DOS "
          f"calc; {counters['no_object']} have a DOS calc whose object is PURGED from MP's "
          f"store (often pre-2021 calcs) — unrecoverable, settled + flagged dos_skip_reason.")
    return counters
