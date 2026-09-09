"""MP discovery screen: push every locally-graphed MP material through saved
transfer heads and rank predicted superconductors (user, 2026-09-09).

    python scripts/screen_mp.py --runs <run_dir> [<run_dir> ...] [--top 40] [--min-tc 15]

For each run dir (a FineTune run with model_seed*.pt) predict_tc.py scores
database/datafiles/MP/screen_mp_index.pickle with descriptors_screen_mp.pickle
(cached per run as predictions_screen_mp.csv). Predictions are merged per
material (one column per encoder, plus their mean and min = the consensus
reading), joined to MP summary metadata, and filtered:
  known    : the material is an MP parent of a row in the SC training index
             (the 2,708 parents) — reported separately, never as a discovery
  chemistry: cuprate-like (Cu+O), nickelate (Ni+O, no Cu), Fe-pnictide/
             chalcogenide (Fe + As/P/Se/Te), hydride (H-rich), other
Writes docs/data_curation/mp_screen_<tag>.csv (everything) and prints the
top candidates overall and per chemistry class with e_above_hull / band gap /
theoretical flags so "notable" can be judged, not just "high".
"""
import argparse
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP = os.path.join(_ROOT, "database", "datafiles", "MP")
INDEX, DESC, META = (os.path.join(MP, f) for f in ("screen_mp_index.pickle", "descriptors_screen_mp.pickle", "screen_mp_metadata.pickle"))


def chem_class(formula):
    els = set(re.findall(r"[A-Z][a-z]?", formula or ""))
    if "Cu" in els and "O" in els:
        return "cuprate-like"
    if "Ni" in els and "O" in els:
        return "nickelate"
    if "Fe" in els and els & {"As", "P", "Se", "Te", "S"}:
        return "Fe-pnictide/chalcogenide"
    if "H" in els:
        m = re.search(r"H(\d+)", formula or "")
        if m and int(m.group(1)) >= 3:
            return "hydride"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--tags", nargs="*", default=None, help="short names per run (default: encoder rung from the run name)")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--min-tc", type=float, default=15.0)
    ap.add_argument("--out-tag", default="v1")
    a = ap.parse_args()
    tags = a.tags or [re.search(r"probe_([0-9a-z]+)", os.path.basename(r)).group(1) for r in a.runs]
    preds = None
    for run, tag in zip(a.runs, tags):
        cache = os.path.join(run, "predictions_screen_mp.csv")
        if not os.path.exists(cache):
            print(f"[screen] scoring {tag} ...", flush=True)
            subprocess.run([sys.executable, os.path.join(_ROOT, "scripts", "predict_tc.py"), run, "--index", INDEX,
                            "--descriptors", DESC, "--out", cache], check=True)
        p = pd.read_csv(cache)[["id", "tc_pred_K", "tc_pred_std_K"]].rename(columns={"tc_pred_K": f"tc_{tag}", "tc_pred_std_K": f"std_{tag}"})
        preds = p if preds is None else preds.merge(p, on="id", how="outer")
    tc_cols = [f"tc_{t}" for t in tags]
    preds["tc_mean"] = preds[tc_cols].mean(axis=1)
    preds["tc_min"] = preds[tc_cols].min(axis=1)
    preds["tc_max"] = preds[tc_cols].max(axis=1)
    meta = pd.read_pickle(META)
    df = preds.merge(meta, on="id", how="left")
    sc = pd.read_pickle(os.path.join(MP, "SC_MP_V4_doped_v45.pickle"))
    known = set(m for i in sc.id for m in re.findall(r"mp-\d+", i))
    df["known_parent"] = df.id.isin(known)
    df["chem"] = df.formula.map(chem_class)
    out = os.path.join(_ROOT, "docs", "data_curation", f"mp_screen_{a.out_tag}.csv")
    df.sort_values("tc_mean", ascending=False).to_csv(out, index=False)
    print(f"[screen] {len(df)} materials scored by {tags}; {int(df.known_parent.sum())} are SC-training parents -> {out}")
    cand = df[~df.known_parent & (df.tc_mean >= a.min_tc)].copy()
    print(f"[screen] {len(cand)} unseen materials with consensus mean >= {a.min_tc} K "
          f"(all encoders >= {a.min_tc}: {int((df[~df.known_parent].tc_min >= a.min_tc).sum())})")
    cols = ["id", "formula", "chem"] + tc_cols + ["tc_mean", "e_above_hull", "band_gap", "theoretical", "spacegroup"]
    fmt = lambda d: d[cols].assign(**{c: d[c].round(1) for c in tc_cols + ["tc_mean"]}, e_above_hull=d.e_above_hull.round(3), band_gap=d.band_gap.round(2)).to_string(index=False)
    print(f"\n=== top {a.top} unseen by consensus mean ===\n" + fmt(cand.sort_values("tc_mean", ascending=False).head(a.top)))
    stable = cand[(cand.e_above_hull <= 0.05) & (cand.band_gap <= 0.1) & (~cand.theoretical.fillna(True))]
    print(f"\n=== of those, experimentally known (not theoretical), on/near hull (<=50 meV) and metallic: {len(stable)} ===\n" + fmt(stable.sort_values("tc_mean", ascending=False).head(a.top)))
    for cls in ("cuprate-like", "nickelate", "Fe-pnictide/chalcogenide", "hydride", "other"):
        sub = cand[cand.chem == cls].sort_values("tc_mean", ascending=False).head(12)
        if len(sub):
            print(f"\n=== {cls}: top {len(sub)} unseen ===\n" + fmt(sub))
    kn = df[df.known_parent].sort_values("tc_mean", ascending=False).head(15)
    print(f"\n=== sanity: highest-predicted KNOWN SC parents (training exposure) ===\n" + fmt(kn))


if __name__ == "__main__":
    main()
