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
_STOCK = {"mp-1"}                               # has an object via the STOCK task-id route (Silicon-like)
_BY_MID = {"mp-3"}                              # has an object only via dos/<mid>.json.gz (mp-1000000-like)
# mp-5: has_props dos but NO object under either scheme -> no_object
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
        FakeMPRester.calls.append(("matid", mid))
        if mid in _BY_MID:
            return ([{"data": _CDos()}], 1)                    # object may be wrapped {"data": dos}
        raise RuntimeError(f"No object found: s3://materialsproject-parsed/dos/{mid}.json.gz")


class _Materials:
    summary = _Summary()
    electronic_structure_dos = _DosRester()


class FakeMPRester:
    calls = []

    def __init__(self, key, use_document_model=True, **kw):
        self.materials = _Materials()

    def get_dos_by_material_id(self, mid):                      # the STOCK task-id route
        FakeMPRester.calls.append(("stock", mid))
        if mid in _STOCK:
            return _CDos()
        # mimic the real "no ES summary doc" error so the fall-through to material-id is tested
        raise RuntimeError(f"No electronic structure data found for material ID {mid}.")

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
        mids = lambda: {m for _, m in FakeMPRester.calls}
        # prefilter path: only HAS materials are fetched. mp-1 resolves via the STOCK task-id
        # route (Silicon-like), mp-3 only via the material-id key (mp-1000000-like), mp-5 via
        # neither -> no_object (settled + flagged).
        FakeMPRester.calls = []
        ip, paths = _make_index(tmp)
        c = D.fetch_and_attach_dos(ip, workers=4)
        dos, miss, purged = _states(paths)
        prefilter = mids() == HAS                                    # only has-DOS materials fetched
        both_routes = (("stock", "mp-1") in FakeMPRester.calls       # stock route recovers Silicon-like
                       and ("matid", "mp-3") in FakeMPRester.calls)  # material-id recovers the newer slice
        coverage = (dos == {"mp-1", "mp-3"} and miss == (ALL - {"mp-1", "mp-3"})
                    and purged == {"mp-5"})                           # object-absent flagged
        counters = (c["ok"] == 2 and c["no_dos"] == 3 and c["no_object"] == 1
                    and c["fail"] == 0)

        # resumability: a second run fetches nothing, all skip.
        FakeMPRester.calls = []
        c2 = D.fetch_and_attach_dos(ip, workers=4)
        resume = (len(FakeMPRester.calls) == 0 and c2["skip"] == 6)

        # fallback: prefilter outage -> attempt all 6; mp-1/mp-3 ok, the rest 404 -> no_object.
        PREFILTER_BREAKS["value"] = True
        FakeMPRester.calls = []
        tmp2 = tempfile.mkdtemp()
        try:
            ip2, paths2 = _make_index(tmp2)
            c3 = D.fetch_and_attach_dos(ip2, workers=4)
            dos2, _, _ = _states(paths2)
            fallback = (mids() == ALL and dos2 == {"mp-1", "mp-3"}
                        and c3["ok"] == 2 and c3["no_object"] == 4)
        finally:
            shutil.rmtree(tmp2)
        PREFILTER_BREAKS["value"] = False

        # retry_no_object: after a fix makes a prior-no_object material's object retrievable,
        # --retry-no-object clears that sentinel and re-attempts ONLY it (mp-0/2/4 stay no_dos).
        _BY_MID.add("mp-5")
        try:
            FakeMPRester.calls = []
            c4 = D.fetch_and_attach_dos(ip, workers=4, retry_no_object=True)
            dos4, _, _ = _states(paths)
            retry = ("mp-5" in dos4 and mids() == {"mp-5"} and c4["ok"] == 1)
        finally:
            _BY_MID.discard("mp-5")

        ok = all([prefilter, both_routes, coverage, counters, resume, fallback, retry])
        print(f"prefilter_skips={prefilter} both_routes(stock+matid)={both_routes} "
              f"coverage={coverage} counters={counters} resumable={resume} fallback={fallback} "
              f"retry_no_object={retry}")
        print("verify_dos_fetch: " + ("PASS" if ok else "FAIL"))
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    sys.exit(main())
