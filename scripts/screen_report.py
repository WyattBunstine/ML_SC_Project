"""Final MP-screen report: ML-head consensus + alpha^2F-head Eliashberg readout.

    python scripts/screen_report.py [--in docs/data_curation/mp_screen_v2.csv] [--tags 51 47] [--top 30]

Tables: (1) the clean ML shortlist (unseen, every ML encoder >= 10 K, metallic,
<= 100 meV above hull) with lambda / omega_log / Tc_AD per e-ph encoder;
(2) the Eliashberg screen: unseen metallic near-hull materials ranked by
Tc_AD(mu*=0.10) of the first e-ph encoder, with the ML consensus alongside;
(3) agreement: materials where BOTH readings clear 10 K. Tc_AD is the
conventional (phonon-mediated) estimate — cuprates/nickelates are expected to
show low Tc_AD, so table 2 is the conventional-superconductor screen.
"""
import argparse
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=os.path.join(_ROOT, "docs", "data_curation", "mp_screen_v2.csv"))
    ap.add_argument("--tags", nargs="+", default=["51", "47"])
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args()
    d = pd.read_csv(a.inp)
    u = d[~d.known_parent].copy()
    metal = (u.band_gap.fillna(9) <= 0.1) & (u.e_above_hull.fillna(9) <= 0.1)
    t0 = a.tags[0]
    lam = lambda t: f"lambda_{t}"; tc = lambda t: f"tcAD_mu010_{t}"; wl = lambda t: f"wlog_K_{t}"
    eph_cols = [c for t in a.tags for c in (lam(t), wl(t), tc(t)) if c in u.columns]
    base = ["id", "formula", "tc_38", "tc_49", "tc_51", "tc_mean"]
    fmt = lambda x: x.assign(**{c: x[c].round(2) if c.startswith("lambda") else x[c].round(1) for c in eph_cols + ["tc_38", "tc_49", "tc_51", "tc_mean"] if c in x}).to_string(index=False)
    print(f"[report] {len(u)} unseen materials; e-ph columns from encoders {a.tags}")
    for t in a.tags:
        if lam(t) in u:
            q = u[lam(t)].quantile([.25, .5, .75, .95]).round(3).tolist()
            print(f"  {t}: lambda quartiles p25/50/75/95 {q} | Tc_AD(0.10) >= 10 K among metallic near-hull unseen: {int((u[metal][tc(t)] >= 10).sum())}")
    clean = u[(u.tc_min >= 10) & metal].sort_values("tc_mean", ascending=False)
    print(f"\n=== (1) clean ML shortlist ({len(clean)}) with the Eliashberg readout ===\n" + fmt(clean[base + eph_cols + ["supercon"]]))
    el = u[metal & (u[tc(t0)] >= 5)].sort_values(tc(t0), ascending=False).head(a.top)
    print(f"\n=== (2) Eliashberg screen: unseen, metallic, near-hull, ranked by Tc_AD(0.10) of encoder {t0} ===\n" + fmt(el[["id", "formula", "chem"] + eph_cols + ["tc_mean", "e_above_hull", "theoretical", "supercon", "supercon_tc_family"]].assign(e_above_hull=el.e_above_hull.round(3))))
    both = u[metal & (u[tc(t0)] >= 10) & (u.tc_min >= 10)].sort_values("tc_mean", ascending=False)
    print(f"\n=== (3) agreement: ML consensus (all >= 10 K) AND Tc_AD({t0}) >= 10 K: {len(both)} ===\n" + (fmt(both[base + eph_cols + ["supercon"]]) if len(both) else "  none"))
    kn = d[d.known_parent & (d.chem != "cuprate-like")].sort_values(tc(t0), ascending=False).head(12)
    print(f"\n=== sanity: known non-cuprate SC parents by Tc_AD({t0}) (training exposure; true Tc in supercon_tc_family) ===\n" + fmt(kn[["id", "formula", "chem"] + eph_cols + ["tc_mean", "supercon_tc_family"]]))


if __name__ == "__main__":
    main()
