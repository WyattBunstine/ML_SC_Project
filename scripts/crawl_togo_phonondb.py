"""Crawl the Togo phonondb per-material zips from MDR@NIMS (task #9).

Registry: the 10,034-row table in github atztogo/phonondb mdr/phonondb/README.md
(MP id -> per-material zip). Downloads zips into
database/datafiles/TogoPhononDB/zips/mp-<id>.zip — resumable (skips existing,
size>0), polite (N workers, courtesy delay). Zips are tiny (phonopy_params.yaml.xz
+ PNGs, ~0.1-2 MB); expected total a few GB. DOS computation from the zips is a
separate step (build_phonon_targets.py, needs phonopy).

  python scripts/crawl_togo_phonondb.py [--workers 3] [--limit N]
"""
import argparse
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "database", "datafiles", "TogoPhononDB")
REG_URL = ("https://raw.githubusercontent.com/atztogo/phonondb/main/"
           "mdr/phonondb/README.md")
ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|[^|]*\|\s*"
                 r"\[[^\]]+\]\((https://mdr\.nims\.go\.jp/download_all/[^)]+\.zip)\)")


def registry():
    path = os.path.join(OUT, "registry.csv")
    if os.path.exists(path):
        rows = [line.rstrip("\n").split(",", 3) for line in open(path)][1:]
        return [(r[0], r[3]) for r in rows]
    txt = urllib.request.urlopen(REG_URL, timeout=120).read().decode()
    rows = []
    for line in txt.splitlines():
        m = ROW.match(line)
        if m:
            rows.append((f"mp-{m.group(1)}", m.group(2), m.group(3), m.group(4)))
    if len(rows) < 1000:
        # A format drift / truncated response must not poison the cache: a
        # 0-row registry.csv would make every later run "succeed" doing nothing.
        raise ValueError(f"registry parse found only {len(rows)} rows "
                         f"(expected ~10,034) — README format changed?")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        f.write("mp_id,formula,spacegroup,url\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"registry: {len(rows)} rows -> {path}", flush=True)
    return [(r[0], r[3]) for r in rows]


def fetch(job):
    mp_id, url = job
    dest = os.path.join(OUT, "zips", f"{mp_id}.zip")
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    try:
        tmp = dest + ".tmp"
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, dest)
        time.sleep(0.3)                       # courtesy
        return "ok"
    except Exception as e:  # noqa: BLE001 — log and move on; rerun retries
        return f"fail {mp_id}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    jobs = registry()
    if a.limit:
        jobs = jobs[: a.limit]
    os.makedirs(os.path.join(OUT, "zips"), exist_ok=True)
    n = {"ok": 0, "skip": 0, "fail": 0}
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(fetch, jobs)):
            n["fail" if res.startswith("fail") else res] += 1
            if res.startswith("fail"):
                print(res, flush=True)
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(jobs)} (ok {n['ok']}, skip {n['skip']}, "
                      f"fail {n['fail']})", flush=True)
    print(f"crawl done: {n}", flush=True)
