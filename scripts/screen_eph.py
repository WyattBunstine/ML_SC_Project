"""Eliashberg-side screen: the pretrained encoders' alpha^2F head -> lambda,
omega_log, omega_2, Allen-Dynes T_c (mu* 0.10 / 0.13) for every material in
the MP screen index, merged into the ML-head screen table (user, 2026-09-09:
"it should cost nothing").

    python scripts/screen_eph.py --checkpoints <ckpt>=<tag> [...] [--in mp_screen_v1.csv] [--out mp_screen_v2.csv]

Reuses the predicted-feature machinery of the pf arms (models/head/
pred_features.py): one frozen forward pass per checkpoint over the screen
index, cached; lambda/omega moments and the Allen-Dynes formula are the ONE
implementation used by the G features. Caveat carried from the pf work: on the
conventional transfer rows Tc_AD vs experiment had MAE 4.9-6.4 K / rho .12-.21
(the a2f head is on-domain-accurate on the Cerqueira metals, brittle off it).
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "models", "common"), os.path.join(_ROOT, "models", "GPSTransformer"),
           os.path.join(_ROOT, "models", "head")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from models.head.FineTune import _encoder  # noqa: E402
from models.head.pred_features import load_or_build_pred_cache, derive_global_features  # noqa: E402

MP = os.path.join(_ROOT, "database", "datafiles", "MP")
INDEX = os.path.join(MP, "screen_mp_index.pickle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="<checkpoint path>=<tag>")
    ap.add_argument("--in", dest="inp", default=os.path.join(_ROOT, "docs", "data_curation", "mp_screen_v1.csv"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "docs", "data_curation", "mp_screen_v2.csv"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    df = pd.read_csv(a.inp)
    for spec in a.checkpoints:
        ckpt, tag = spec.rsplit("=", 1)
        print(f"[eph] {tag}: {ckpt}", flush=True)
        model, ds, collate = _encoder(ckpt, INDEX, device)
        cache = load_or_build_pred_cache(model, ds, collate, device, ckpt, INDEX,
                                         cache_path=os.path.join(MP, f"pred_cache_screen_{tag}.pt"))
        names, gmap = derive_global_features(cache)
        ix = {n: i for i, n in enumerate(names)}
        print(f"[eph] global feature names: {names}", flush=True)
        col = {k: next(n for n in names if k in n) for k in ("lambda", "logwlog", "logw2")}
        tcad = [n for n in names if "tcad" in n.lower() or "tc_ad" in n.lower() or "allen" in n.lower()]
        rows = []
        for cid, v in gmap.items():
            v = np.asarray(v, dtype=float)
            mid = cid[:-4] if cid.endswith(".cif") else cid
            r = {"id": mid, f"lambda_{tag}": v[ix[col["lambda"]]], f"wlog_K_{tag}": math.expm1(v[ix[col["logwlog"]]]),
                 f"w2_K_{tag}": math.expm1(v[ix[col["logw2"]]])}
            for n in tcad:
                mu = "010" if "10" in n[-3:] or n.endswith("0.10") else "013"
                r[f"tcAD_mu{mu}_{tag}"] = math.expm1(v[ix[n]])
            rows.append(r)
        e = pd.DataFrame(rows).drop_duplicates("id")
        df = df.merge(e, on="id", how="left")
        lam = e[f"lambda_{tag}"]
        print(f"[eph] {tag}: {len(e)} materials | lambda quartiles p25/50/75 {lam.quantile(.25):.3f}/{lam.quantile(.5):.3f}/{lam.quantile(.75):.3f} | "
              f"Tc_AD(0.10) >= 10 K: {int((e[[c for c in e.columns if 'tcAD_mu010' in c][0]] >= 10).sum())}", flush=True)
        del model, ds, cache
        torch.cuda.empty_cache()
    df.to_csv(a.out, index=False)
    print(f"[eph] wrote {a.out} ({len(df)} rows)", flush=True)


if __name__ == "__main__":
    main()
