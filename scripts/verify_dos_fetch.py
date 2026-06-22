#!/usr/bin/env python3
"""Verify the parallelized fetch-dos orchestration WITHOUT network: a mock MPRester
exercises the bulk has-DOS pre-filter, the thread-pool fetch, the MATERIAL-ID-keyed DOS
download (dos/<mid>.json.gz — the actual open-data layout), the no_object outcome when a
material's DOS object is absent, resumability, and the no-prefilter fallback.

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

HAS = {"mp-1", "mp-3", "mp-5"}                 # has_props says these have a DOS calc
_STORED = {"mp-1", "mp-3"}                      # ...but only these have an OBJECT at dos/<mid>.json.gz
PREFILTER_BREAKS = {"value": False}


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


class _DosRester:
    def _query_open_data(self, bucket=None, key=None, decoder=None):
        mid = key.split("/")[-1].replace(".json.gz", "")       # dos/<mid>.json.gz -> mid
        FakeMPRester.dos_obj_calls.append(mid)
        if mid in _STORED:
            return ([{"data": _CDos()}], 1)                    # object may be wrapped {"data": dos}
        raise RuntimeError(f"No object found: s3://materialsproject-parsed/dos/{mid}.json.gz")


class _Materials:
    summary = _Summary()
    electronic_structure_dos = _DosRester()


class FakeMPRester:
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
    dos, miss, purged = set(), set(), set()
    for p in paths:
        g = json.load(open(p))
        mid = os.path.basename(p).replace(".json", "")
        if "dos" in g:
            dos.add(mid)
            assert len(g["dos"]) == D.N_ENERGY, f"bad dos len for {mid}"
        if g.get("dos_missing"):
            miss.add(mid)
        if g.get("dos_skip_reason") == "no_object":
            purged.add(mid)
    return dos, miss, purged


def main():
    tmp = tempfile.mkdtemp()
    try:
        ALL = {f"mp-{i}" for i in range(6)}
        # prefilter path: only HAS materials are downloaded by material-id key; mp-1/mp-3 have
        # objects -> ok; mp-5 has a DOS calc but no object -> no_object (settled + flagged).
        FakeMPRester.dos_obj_calls = []
        ip, paths = _make_index(tmp)
        c = D.fetch_and_attach_dos(ip, workers=4)
        dos, miss, purged = _states(paths)
        prefilter = set(FakeMPRester.dos_obj_calls) == HAS           # only has-DOS materials downloaded
        byid = all(k.startswith("mp-") for k in FakeMPRester.dos_obj_calls)  # keyed by material id
        coverage = (dos == {"mp-1", "mp-3"} and miss == (ALL - {"mp-1", "mp-3"})
                    and purged == {"mp-5"})                           # object-absent flagged
        counters = (c["ok"] == 2 and c["no_dos"] == 3 and c["no_object"] == 1
                    and c["fail"] == 0)

        # resumability: a second run downloads nothing, all skip.
        FakeMPRester.dos_obj_calls = []
        c2 = D.fetch_and_attach_dos(ip, workers=4)
        resume = (len(FakeMPRester.dos_obj_calls) == 0 and c2["skip"] == 6)

        # fallback: prefilter outage -> attempt all 6 by material-id; mp-1/mp-3 ok, rest 404.
        PREFILTER_BREAKS["value"] = True
        FakeMPRester.dos_obj_calls = []
        tmp2 = tempfile.mkdtemp()
        try:
            ip2, paths2 = _make_index(tmp2)
            c3 = D.fetch_and_attach_dos(ip2, workers=4)
            dos2, _, _ = _states(paths2)
            fallback = (set(FakeMPRester.dos_obj_calls) == ALL and dos2 == {"mp-1", "mp-3"}
                        and c3["ok"] == 2 and c3["no_object"] == 4)
        finally:
            shutil.rmtree(tmp2)
        PREFILTER_BREAKS["value"] = False

        # retry_no_object: after a fix makes a prior-no_object material's object retrievable,
        # --retry-no-object clears that sentinel and re-attempts ONLY it (mp-0/2/4 stay no_dos).
        _STORED.add("mp-5")
        try:
            FakeMPRester.dos_obj_calls = []
            c4 = D.fetch_and_attach_dos(ip, workers=4, retry_no_object=True)
            dos4, _, _ = _states(paths)
            retry = ("mp-5" in dos4 and FakeMPRester.dos_obj_calls == ["mp-5"]
                     and c4["ok"] == 1)
        finally:
            _STORED.discard("mp-5")

        ok = all([prefilter, byid, coverage, counters, resume, fallback, retry])
        print(f"prefilter_skips={prefilter} keyed_by_material_id={byid} coverage={coverage} "
              f"counters={counters} resumable={resume} fallback={fallback} retry_no_object={retry}")
        print("verify_dos_fetch: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(main())
