#!/usr/bin/env python3
"""Verify the parallelized + schema-robust fetch-dos orchestration WITHOUT network: a
mock MPRester exercises the bulk has-DOS pre-filter, the thread-pool fetch, the RAW-mode
DOS retrieval that survives the emmet-core/server task_id mismatch (the ValidationError
storm), resumability, and the no-prefilter fallback.

    python scripts/verify_dos_fetch.py     # exit 0 = pass, 1 = fail, 77 = skipped
"""
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "database"))
os.environ.setdefault("MP_API_KEY", "test-key")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import mp_api.client as mpc  # noqa: E402
except Exception as exc:  # noqa: BLE001 — DOS fetch is optional infra
    print(f"verify_dos_fetch: SKIP (mp_api not importable: {type(exc).__name__})")
    sys.exit(77)

import Download_MP_dos as D  # noqa: E402

HAS = {"mp-1", "mp-3", "mp-5"}                 # has_props says these have a DOS
PREFILTER_BREAKS = {"value": False}


def _summary_for(mid):
    """Raw ES 'dos' summary per material, modelling the schema drift:
      mp-1: task_id in total.1 (the happy path)
      mp-3: total.1 MISSING task_id, but total.-1 has it -> the walk must recover it
      mp-5: task_id absent everywhere -> unrecoverable (no usable DOS)"""
    if mid == "mp-1":
        return {"total": {"1": {"band_gap": 1.0, "task_id": "mp-1-dos"}}}
    if mid == "mp-3":
        return {"total": {"1": {"band_gap": 2.0},
                          "-1": {"band_gap": 2.0, "task_id": "mp-3-dos"}},
                "elemental": {"Fe": {"s": {"1": {"band_gap": 2.0}}}}}
    if mid == "mp-5":
        return {"total": {"1": {"band_gap": 3.0}, "-1": {"band_gap": 3.0}}}
    return None


class _CDos:
    def __init__(self):
        self.energies = np.linspace(-12.0, 7.0, 400)
        self.densities = {1: np.abs(np.sin(self.energies)) + 0.05}
        self.efermi = 0.0


class _Doc:
    def __init__(self, mid):
        self.material_id = mid


class _Summary:
    def search(self, material_ids=None, has_props=None, fields=None):
        if PREFILTER_BREAKS["value"]:
            raise RuntimeError("simulated prefilter outage")
        return [_Doc(m) for m in material_ids if m in HAS]      # server-side has_props filter


class _ESRester:
    def search(self, material_ids=None, fields=None):
        FakeMPRester.es_search_calls.append(material_ids)      # mid is a single string here
        s = _summary_for(material_ids)
        return [{"dos": s}] if s is not None else []


class _DosRester:
    es_rester = _ESRester()

    def get_dos_from_task_id(self, tid):
        FakeMPRester.dos_obj_calls.append(tid)
        return _CDos()


class _Materials:
    summary = _Summary()
    electronic_structure_dos = _DosRester()


class FakeMPRester:
    es_search_calls = []
    dos_obj_calls = []

    def __init__(self, key, use_document_model=True, **kw):
        self.materials = _Materials()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


mpc.MPRester = FakeMPRester


def _make_index(tmp, n=6):
    gd = os.path.join(tmp, "graphs")
    os.makedirs(gd)
    ids, paths = [], []
    for i in range(n):
        gp = os.path.join(gd, f"mp-{i}.json")
        json.dump({"nodes": [{"Z": 11}], "edges": []}, open(gp, "w"))
        ids.append(f"mp-{i}")
        paths.append(gp)
    ip = os.path.join(tmp, "idx.pickle")
    pd.DataFrame({"id": ids, "graph_path": paths}).to_pickle(ip)
    return ip, paths


def _states(paths):
    dos, miss = set(), set()
    for p in paths:
        g = json.load(open(p))
        mid = os.path.basename(p).replace(".json", "")
        if "dos" in g:
            dos.add(mid)
            assert len(g["dos"]) == D.N_ENERGY, f"bad dos len for {mid}"
        if g.get("dos_missing"):
            miss.add(mid)
    return dos, miss


def main():
    tmp = tempfile.mkdtemp()
    try:
        ALL = {f"mp-{i}" for i in range(6)}
        # prefilter path: only HAS materials hit the ES endpoint; task_id walk recovers
        # mp-3 (task_id only in total.-1); mp-5 has none -> no_dos.
        FakeMPRester.es_search_calls = []
        FakeMPRester.dos_obj_calls = []
        ip, paths = _make_index(tmp)
        c = D.fetch_and_attach_dos(ip, workers=4)
        dos, miss = _states(paths)
        prefilter = set(FakeMPRester.es_search_calls) == HAS         # no ES call for no-DOS majority
        recover = set(FakeMPRester.dos_obj_calls) == {"mp-1-dos", "mp-3-dos"}  # drift recovered
        coverage = dos == {"mp-1", "mp-3"} and miss == (ALL - {"mp-1", "mp-3"})
        counters = c["ok"] == 2 and c["no_dos"] == 4 and c["fail"] == 0

        # resumability: a second run hits the ES endpoint zero times, all skip.
        FakeMPRester.es_search_calls = []
        c2 = D.fetch_and_attach_dos(ip, workers=4)
        resume = (len(FakeMPRester.es_search_calls) == 0 and c2["skip"] == 6)

        # fallback: prefilter outage -> query all 6 at the ES endpoint, still correct.
        PREFILTER_BREAKS["value"] = True
        FakeMPRester.es_search_calls = []
        tmp2 = tempfile.mkdtemp()
        try:
            ip2, paths2 = _make_index(tmp2)
            c3 = D.fetch_and_attach_dos(ip2, workers=4)
            dos2, _ = _states(paths2)
            fallback = (set(FakeMPRester.es_search_calls) == ALL
                        and dos2 == {"mp-1", "mp-3"} and c3["ok"] == 2 and c3["no_dos"] == 4)
        finally:
            shutil.rmtree(tmp2)
        PREFILTER_BREAKS["value"] = False

        ok = all([prefilter, recover, coverage, counters, resume, fallback])
        print(f"prefilter_skips={prefilter} taskid_walk_recovers={recover} coverage={coverage} "
              f"counters={counters} resumable={resume} fallback={fallback}")
        print("verify_dos_fetch: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(main())
