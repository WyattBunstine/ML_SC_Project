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


def _is_terminal_fail(exc):
    """A deterministic, non-retryable failure that is NOT 'no DOS': a schema ValidationError,
    or a 'No object found' from the DOS object store (the task_id we resolved doesn't key a
    stored DOS — an mp-api/emmet-core vs MP-layout lag). Terminal so we don't retry 3x, but
    counted as a FAIL (left un-marked) rather than no_dos, so a future client fix re-tries it."""
    return _is_validation_error(exc) or "no object found" in str(exc).lower()


def _first_task_id(obj):
    """First non-empty `task_id` anywhere in a (raw) DOS summary dict. The summary nests
    task_id under total/elemental/orbital per spin; ALL entries reference the SAME DOS calc,
    so any one is the id we download by. Walking the raw dict (vs the stock
    dos["total"]["1"]["task_id"]) survives the emmet-core/server schema drift where some
    entries arrive without task_id (the cause of the ValidationError storm)."""
    if isinstance(obj, dict):
        tid = obj.get("task_id")
        if isinstance(tid, str) and tid:
            return tid
        for v in obj.values():
            r = _first_task_id(v)
            if r:
                return r
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            r = _first_task_id(v)
            if r:
                return r
    return None


def _download_dos(dr, task_id):
    """Download a DOS CompleteDos by task_id, RAW S3 key first. The stock
    get_dos_from_task_id runs the id through validate_ids, which currently normalizes a
    'blessed' AlphaID (e.g. 'aaadtbme') to a legacy numeric id ('1705864') whose object
    isn't stored -> 'No object found'. The object lives under the AlphaID key, so we fetch
    `dos/<raw id>.json.gz` directly (same bucket/decoder as the stock method) and only fall
    back to the validated path if the raw key is genuinely absent (covers already-numeric ids)."""
    from mp_api.client.core.utils import load_json
    try:
        res = dr._query_open_data(
            bucket="materialsproject-parsed",
            key=f"dos/{task_id}.json.gz",
            decoder=lambda x: load_json(x, deser=True))
        return res[0][0]["data"]
    except Exception as exc:  # noqa: BLE001
        if "no object found" not in str(exc).lower():
            raise
        return dr.get_dos_from_task_id(task_id)          # fallback: validated/normalized id


def _dos_object(mpr, mid):
    """CompleteDos for one material, ROBUST to the emmet-core 0.86.x / server mismatch: the ES
    'dos' summary omits the (required) per-entry task_id (so the stock get_dos_by_material_id
    raises a ValidationError) AND moved the id to an AlphaID that validate_ids mis-normalizes.
    The client is RAW mode (use_document_model=False) so the dos-summary query skips
    validation; we take the DOS task_id from anywhere in the summary and download by its RAW
    key. Returns None when the material has no DOS doc or no recoverable task_id."""
    dr = mpr.materials.electronic_structure_dos          # DosRester (raw mode -> raw es_rester)
    docs = dr.es_rester.search(material_ids=mid, fields=["dos"])
    if not docs:
        return None
    doc0 = docs[0]
    summary = doc0.get("dos") if isinstance(doc0, dict) else getattr(doc0, "dos", None)
    task_id = _first_task_id(summary)
    if not task_id:
        return None
    return _download_dos(dr, task_id)


def diagnose_dos(api_key, mid):
    """Print the RAW ES 'dos' summary for one material id and whether a task_id is
    recoverable — to confirm the schema and the fix on a real failing material:
        python main.py fetch-dos --index <any> --diagnose mp-11944"""
    import json as _json
    from mp_api.client import MPRester
    with MPRester(api_key, use_document_model=False) as mpr:
        dr = mpr.materials.electronic_structure_dos
        # Collect every candidate DOS task_id we can reach, then test each against the
        # object store — to find which (if any) source yields a real DOS for this material.
        candidates = []                                  # [(source_label, task_id)]
        docs = dr.es_rester.search(material_ids=mid, fields=["dos"])
        summary = (docs[0].get("dos") if docs and isinstance(docs[0], dict) else None)
        print(f"  {mid}: raw dos summary =\n{_json.dumps(summary, indent=2, default=str)[:1200]}")
        top = _first_task_id(summary)
        if top:
            candidates.append(("dos-summary", top))
        try:                                             # provenance — may name the real DOS calc
            sdocs = mpr.materials.summary.search(material_ids=mid, fields=["origins"])
            origins = ((sdocs[0].get("origins") if sdocs and isinstance(sdocs[0], dict) else None)
                       or [])
            print(f"  origins = {_json.dumps(origins, default=str)[:700]}")
            for o in origins:
                if isinstance(o, dict) and o.get("task_id"):
                    candidates.append((f"origin:{o.get('name')}", str(o["task_id"])))
        except Exception as exc:  # noqa: BLE001
            print(f"  origins probe failed: {type(exc).__name__}: {str(exc)[:120]}")

        if not candidates:
            print("  -> no candidate task_id anywhere (settles no_dos).")
            return

        # Surface the REAL S3 error: _query_open_data collapses every botocore ClientError
        # (AccessDenied / NoSuchKey / region / unsigned-access) into a generic 'No object
        # found', so a bucket-access misconfig looks identical to a purged object. Probe the
        # bucket directly and report the actual error code — the discriminator for ok=0.
        from io import BytesIO
        def _probe_s3(tid):
            key = f"dos/{tid}.json.gz"
            try:
                dr.s3_client.download_fileobj("materialsproject-parsed", key, BytesIO())
                return f"S3 OK (object exists at {key})"
            except Exception as exc:  # noqa: BLE001
                resp = getattr(exc, "response", None)
                if isinstance(resp, dict):
                    e = resp.get("Error", {})
                    return f"S3 {e.get('Code', '?')}: {str(e.get('Message', ''))[:90]} [{key}]"
                return f"S3 {type(exc).__name__}: {str(exc)[:90]} [{key}]"

        def _probe(method, tid):
            try:
                grid = dos_to_grid(*complete_dos_total(method(tid)))
                return (f"OK grid[{len(grid)}] sum={float(grid.sum()):.3f}"
                        if grid is not None else "downloaded but degenerate")
            except Exception as exc:  # noqa: BLE001
                return f"FAILED {type(exc).__name__}: {str(exc)[:80]}"

        print(f"  testing {len(candidates)} candidate task_id(s):")
        for label, tid in candidates:
            print(f"    [{label}] {tid}:")
            print(f"        raw S3 probe -> {_probe_s3(tid)}")            # true botocore error code
            print(f"        raw-key dl   -> {_probe(lambda t: _download_dos(dr, t), tid)}")
        print("  (S3 'NoSuchKey' = object truly purged; 'AccessDenied'/'InvalidAccessKeyId'/"
              "region/signature = a bucket-access misconfig -> the real ok=0 cause. Paste back.)")


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


def fetch_and_attach_dos(index_path, broaden_ev=0.1, limit=None, retries=3, workers=8):
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
        if "dos" in graph or graph.get("dos_missing"):      # resumable: hit OR known miss
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
