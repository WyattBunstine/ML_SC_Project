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
    # ---- magnetism gate (2026-09-10): the a2f head assigns lambda > 1 to Mn/Cr/Eu
    # compounds it has never seen as non-superconducting. Gate = MP says non-
    # magnetic (flag AND |total magnetization| < 0.1 mu_B) AND every e-ph encoder
    # predicts mean |m| < 0.1 mu_B/atom (its own magmom head; catches Mn2P /
    # Eu3As4 / MnO / NiO that MP's flag misses). The staggered channel is NOT
    # used: it reads 0.7-0.9 on non-magnetic Pd2B / YWC2 / TcN.
    mabs_cols = [f"mabs_{t}" for t in a.tags if f"mabs_{t}" in u.columns]
    nonmag = (u.is_magnetic.fillna(True).astype(str).str.lower().eq("false")
              & (u.magnetization.fillna(9).abs() < 0.1)
              & (u[mabs_cols].max(axis=1) < 0.1))
    print(f"\n[gate] unseen metallic near-hull: {int(metal.sum())} | after magnetism gate: {int((metal & nonmag).sum())} "
          f"| gate removes {int((metal & ~nonmag & (u[tc(t0)] >= 20)).sum())} of {int((metal & (u[tc(t0)] >= 20)).sum())} rows with Tc_AD({t0}) >= 20 K")
    # ---- scale calibration on KNOWN conventional parents (training exposure): median true/predicted ----
    kn_all = d[d.known_parent & (d.chem != "cuprate-like") & (d.supercon_tc_family > 1)]
    for t in a.tags:
        ok = kn_all[kn_all[tc(t)] > 1]
        ratio = (ok.supercon_tc_family / ok[tc(t)])
        print(f"[calib] {t}: on {len(ok)} known non-cuprate SC parents with Tc_AD > 1 K, median true/Tc_AD = {ratio.median():.2f} "
              f"(p25 {ratio.quantile(.25):.2f}, p75 {ratio.quantile(.75):.2f}); spearman(Tc_AD, true) = {ok[tc(t)].corr(ok.supercon_tc_family, method='spearman'):.2f}")
    el = u[metal & nonmag & (u[tc(t0)] >= 5)].sort_values(tc(t0), ascending=False).head(a.top)
    print(f"\n=== (2) Eliashberg screen, GATED: unseen, metallic, near-hull, non-magnetic, ranked by Tc_AD(0.10) of encoder {t0} ===\n" + fmt(el[["id", "formula", "chem"] + eph_cols + mabs_cols + ["tc_mean", "e_above_hull", "theoretical", "supercon", "supercon_tc_family"]].assign(e_above_hull=el.e_above_hull.round(3), **{c: el[c].round(2) for c in mabs_cols})))
    both = u[metal & nonmag & (u[tc(t0)] >= 10) & (u.tc_min >= 10)].sort_values("tc_mean", ascending=False)
    print(f"\n=== (3) agreement (gated): ML consensus (all >= 10 K) AND Tc_AD({t0}) >= 10 K: {len(both)} ===\n" + (fmt(both[base + eph_cols + ["supercon"]]) if len(both) else "  none"))
    kn = d[d.known_parent & (d.chem != "cuprate-like")].sort_values(tc(t0), ascending=False).head(12)
    print(f"\n=== sanity: known non-cuprate SC parents by Tc_AD({t0}) (training exposure; true Tc in supercon_tc_family) ===\n" + fmt(kn[["id", "formula", "chem"] + eph_cols + ["tc_mean", "supercon_tc_family"]]))


if __name__ == "__main__":
    main()
