#!/usr/bin/env python3
"""Verify the parallelized fetch-dos orchestration WITHOUT network: a mock MPRester
exercises the bulk has-DOS pre-filter, the thread-pool fetch, resumability, the
sentinel writes, and the no-prefilter fallback. (The grid math has its own coverage;
this guards the concurrency/pre-filter/resume logic in database/Download_MP_dos.py.)

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

HAS = {"mp-1", "mp-3", "mp-5"}                 # which materials "have" a DOS
PREFILTER_BREAKS = {"value": False}            # toggle to exercise the fallback


class _Doc:
    def __init__(self, mid, has=None):
        self.material_id = mid
        if has is not None:
            self.has_props = ["dos"] if has else ["materials", "thermo"]


class _CDos:
    def __init__(self):
        self.energies = np.linspace(-12.0, 7.0, 400)
        self.densities = {1: np.abs(np.sin(self.energies)) + 0.05}
        self.efermi = 0.0


class _Summary:
    def search(self, material_ids=None, has_props=None, fields=None):
        if PREFILTER_BREAKS["value"]:
            raise RuntimeError("simulated prefilter outage")
        if has_props and "dos" in has_props:
            return [_Doc(m) for m in material_ids if m in HAS]
        return [_Doc(m, has=(m in HAS)) for m in material_ids]


class _Materials:
    summary = _Summary()


class FakeMPRester:
    get_dos_calls = []

    def __init__(self, key):
        self.materials = _Materials()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_dos_by_material_id(self, mid):
        FakeMPRester.get_dos_calls.append(mid)
        if mid in HAS:
            return _CDos()
        raise RuntimeError("404 not found: no DOS for this material")


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
        # prefilter path: only HAS materials get a get_dos download.
        FakeMPRester.get_dos_calls = []
        ip, paths = _make_index(tmp)
        c = D.fetch_and_attach_dos(ip, workers=4)
        dos, miss = _states(paths)
        prefilter = set(FakeMPRester.get_dos_calls) == HAS
        coverage = dos == HAS and miss == ({f"mp-{i}" for i in range(6)} - HAS)
        counters = c["ok"] == 3 and c["no_dos"] == 3 and c["fail"] == 0

        # resumability: a second run does ZERO downloads, all skip.
        FakeMPRester.get_dos_calls = []
        c2 = D.fetch_and_attach_dos(ip, workers=4)
        resume = (len(FakeMPRester.get_dos_calls) == 0 and c2["skip"] == 6)

        # fallback: prefilter outage -> fetch all in parallel, misses settle via 404.
        PREFILTER_BREAKS["value"] = True
        FakeMPRester.get_dos_calls = []
        tmp2 = tempfile.mkdtemp()
        try:
            ip2, paths2 = _make_index(tmp2)
            c3 = D.fetch_and_attach_dos(ip2, workers=4)
            dos2, _ = _states(paths2)
            fallback = (set(FakeMPRester.get_dos_calls) == {f"mp-{i}" for i in range(6)}
                        and dos2 == HAS and c3["ok"] == 3 and c3["no_dos"] == 3)
        finally:
            shutil.rmtree(tmp2)
        PREFILTER_BREAKS["value"] = False

        ok = all([prefilter, coverage, counters, resume, fallback])
        print(f"prefilter_skips_downloads={prefilter} coverage={coverage} counters={counters} "
              f"resumable={resume} fallback={fallback}")
        print("verify_dos_fetch: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(main())
