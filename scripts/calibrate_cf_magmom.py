"""Calibrate the AOM crystal-field knobs against MPtrj DFT site magmoms.

The AOM feature (RPToleranceFactor/crystal_field_aom.py) predicts per-site
unpaired d-electron counts; DFT site magmoms are the ground truth (spin-only:
|m| ~ n_unpaired for 3d, systematically shrunk ~10-20% by covalency). This
script:

  1. streams a sample of MPtrj materials (first spin-polarized frame each,
     restricted to d-block-containing structures) from the 12 GB release JSON;
  2. runs the RPToleranceFactor graph builder ONCE per structure (oxidation
     guess + Voronoi/ECoN edges are knob-independent) and caches, per TM site,
     the AOM inputs (species+oxidation, neighbor vecs/weights/ligands) plus the
     |DFT magmom| target;
  3. grid-sweeps (BETA_NEPHELAUXETIC, K_SD, 4d/5d row factors) at the fixed
     physical e_sigma scale (the e_sigma x beta grid is ratio-degenerate) over
     the cache, fitting on a train split of materials,
     scoring MAE(n_unpaired_pred, |m_DFT|) plus a spin-STATE agreement rate on
     the discriminative d4-d7 sites (HS/LS separated by |m|: LS < 1.5 < HS);
  4. reports the train sweep, the HELD-OUT verdict (30% material-split test —
     the knobs must generalize, not memorize the sample), and per-element
     diagnostics on test.

This is the GO/NO-GO gate before any full pack rebuild (task #19 stage 4).

  PYTHONHASHSEED=0 python scripts/calibrate_cf_magmom.py --n-materials 1500
"""
import argparse
import os
import pickle
import random
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "database"))
import crystal_graph_v4_import  # noqa: F401,E402  (puts RPToleranceFactor on the path)
import crystal_field_aom as cfa  # noqa: E402
from crystal_graph_v4 import build_crystal_graph_from_structure  # noqa: E402

_TM = set(cfa._D_GROUP)  # d-block symbols


def stream_sample(json_path, n_materials, seed=0, keep_p=0.25):
    """First magmom-carrying frame of thinned materials until n collected.
    Seeded thinning spreads the sample across the whole release file (mp-ids
    are ordered, so an unthinned prefix would bias toward early materials)."""
    import ijson
    from pymatgen.core import Structure
    rng = random.Random(seed)
    out = 0
    with open(json_path, "rb") as f:
        for mp_id, frames in ijson.kvitems(f, "", use_float=True):
            if out >= n_materials:
                break
            if not isinstance(frames, dict) or rng.random() > keep_p:
                continue
            for _fid, frame in frames.items():
                st_d, m = frame.get("structure"), frame.get("magmom")
                if not st_d or m is None:
                    continue
                try:
                    st = Structure.from_dict(st_d)
                except Exception:  # noqa: BLE001
                    break
                if len(st) != len(m) or len(st) > 64:
                    break
                if not any(sp.symbol in _TM for sp in st.composition):
                    break
                yield mp_id, st, np.abs(np.asarray(m, float))
                out += 1
                break  # one frame per material


def _build_one(rec):
    """Worker: structure dict + |m| -> per-TM-site AOM inputs (or None).
    Mirrors the builder's ANION GATE: only anion-role neighbors exert a field."""
    from pymatgen.core import Structure
    mp_id, st_d, mabs = rec
    try:
        g = build_crystal_graph_from_structure(Structure.from_dict(st_d))
    except Exception:  # noqa: BLE001
        return None
    nodes, edges = g["nodes"], g["edges"]
    nbrs = [[] for _ in nodes]
    for e in edges:
        s, t = e["source"], e["target"]
        v = np.asarray(e["cart_vec"], float)
        if nodes[t]["ion_role"] == "anion":
            nbrs[s].append({"vec": v, "weight": float(e["ecn_weight_source"]),
                            "ligand": nodes[t]["element"], "chi": nodes[t]["chi_pauling"]})
        if nodes[s]["ion_role"] == "anion":
            nbrs[t].append({"vec": -v, "weight": float(e["ecn_weight_target"]),
                            "ligand": nodes[s]["element"], "chi": nodes[s]["chi_pauling"]})
    out = []
    for i, n in enumerate(nodes):
        if n["element"] not in _TM:
            continue
        oxi = float(n["oxidation_state"])
        n_d = cfa.d_electron_count(n["element"], oxi)
        if n_d is None or n_d <= 1e-6 or n_d >= 10 - 1e-6:
            continue  # only partially-filled d shells are informative
        out.append({"mp_id": mp_id, "el": n["element"], "oxi": oxi,
                    "n_d": n_d, "species": n["species"], "nbrs": nbrs[i],
                    "m": float(mabs[i])})
    return out


def gather_sites(json_path, n_materials, cache_path, workers=None, keep_p=0.25):
    """Two-phase: stream raw records (IO-bound), then build in parallel."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    from pymatgen.core import Structure  # noqa: F401  (import check before fork)
    recs = []
    for mp_id, st, mabs in stream_sample(json_path, n_materials, keep_p=keep_p):
        recs.append((mp_id, st.as_dict(), mabs))
    print(f"streamed {len(recs)} candidate structures; building graphs...", flush=True)
    import multiprocessing as mp
    workers = workers or max(1, (os.cpu_count() or 4) - 2)
    sites, n_struct, n_fail = [], 0, 0
    with mp.Pool(workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_build_one, recs, chunksize=4)):
            if res is None:
                n_fail += 1
            else:
                n_struct += 1
                sites.extend(res)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(recs)} -> {len(sites)} TM sites "
                      f"({n_fail} failures)", flush=True)
    blob = {"sites": sites, "n_struct": n_struct, "n_fail": n_fail}
    with open(cache_path, "wb") as f:
        pickle.dump(blob, f)
    return blob


_SITES = None  # worker-shared via fork (copy-on-write)


def _set_knobs(beta, k_sd, row4, row5):
    cfa.E_SIGMA_SCALE = 1.0          # physical absolute scale; beta carries the ratio
    cfa.BETA_NEPHELAUXETIC = beta
    cfa.K_SD = k_sd
    cfa.ROW_ESIGMA_FACTOR = {"3d": 1.0, "4d": row4, "5d": row5}


def _score_combo(combo):
    """Score one knob combo over the module-global _SITES."""
    beta, k_sd, row4, row5 = combo
    _set_knobs(beta, k_sd, row4, row5)
    err = n = st_ok = st_tot = 0
    for s in _SITES:
        pred = (cfa.site_cf_features(s["species"], s["nbrs"],
                                     default_oxidation=s["oxi"])["cf_unpaired"]
                if s["nbrs"] else 0.0)
        err += abs(pred - s["m"])
        n += 1
        if 4 <= round(s["n_d"]) <= 7:
            st_tot += 1
            st_ok += int((s["m"] > 1.5) == (pred > 1.5))
    return err / max(n, 1), (st_ok / st_tot if st_tot else float("nan")), combo


def _eval(sites, combo):
    global _SITES
    keep = _SITES
    _SITES = sites
    out = _score_combo(combo)
    _SITES = keep
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="database/datafiles/MPtrj/MPtrj_2022.9_full.json")
    ap.add_argument("--n-materials", type=int, default=10000)
    ap.add_argument("--keep-p", type=float, default=0.5)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--cache", default="model_data/cf_calib/sites_10k.pickle")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    blob = gather_sites(args.json, args.n_materials, args.cache,
                        workers=args.workers, keep_p=args.keep_p)
    sites = blob["sites"]
    # material-level held-out split (sites of one structure are correlated):
    # knobs are FIT on train and REPORTED on test, so the calibration verdict
    # is about generalization, not sample memorization.
    rng = random.Random(123)
    mp_ids = sorted({s["mp_id"] for s in sites})
    rng.shuffle(mp_ids)
    n_test = int(len(mp_ids) * args.test_frac)
    test_ids = set(mp_ids[:n_test])
    train = [s for s in sites if s["mp_id"] not in test_ids]
    test = [s for s in sites if s["mp_id"] in test_ids]
    print(f"\ncalibration pool: {len(sites)} TM sites / {len(mp_ids)} materials "
          f"({blob['n_fail']} build failures) -> train {len(train)} / test {len(test)} sites")
    m_all = np.array([s["m"] for s in sites])
    print(f"|magmom| distribution: {np.percentile(m_all, [5, 25, 50, 75, 95]).round(2)} "
          f"(frac >1.5: {(m_all > 1.5).mean():.2f})")

    grid = [(b, k, r4, r5)
            for b in (0.35, 0.45, 0.51, 0.60, 0.75, 0.90)
            for k in (0.0, 1.0, 2.0)
            for (r4, r5) in ((1.3, 1.6), (1.6, 2.0), (2.0, 2.5))]
    import multiprocessing as mp
    global _SITES
    _SITES = train
    workers = args.workers or max(1, (os.cpu_count() or 4) - 2)

    def _init_sites(s):
        global _SITES
        _SITES = s
    # explicit initializer: portable across start methods (spawn/forkserver
    # workers re-import the module and would see _SITES=None; review 2026-07-28)
    with mp.Pool(workers, initializer=_init_sites, initargs=(train,)) as pool:
        rows = sorted(pool.map(_score_combo, grid))
    print(f"\n=== TRAIN sweep ({len(train)} sites, e_scale=1.0 fixed) ===")
    print(f"{'MAE':>7} {'HS/LS':>6} {'beta':>5} {'k_sd':>5} {'r4d':>5} {'r5d':>5}")
    for mae, acc, (b, k, r4, r5) in rows[:8]:
        print(f"{mae:7.3f} {acc:6.1%} {b:5.2f} {k:5.1f} {r4:5.2f} {r5:5.2f}")
    print("    ...")
    for mae, acc, (b, k, r4, r5) in rows[-2:]:
        print(f"{mae:7.3f} {acc:6.1%} {b:5.2f} {k:5.1f} {r4:5.2f} {r5:5.2f}")

    best = rows[0][2]
    tr_mae, tr_acc, _ = _eval(train, best)
    te_mae, te_acc, _ = _eval(test, best)
    base = float(np.mean([s["m"] for s in test]))
    print(f"\n=== HELD-OUT verdict (best knobs beta={best[0]}, k_sd={best[1]}, "
          f"row={best[2]}/{best[3]}) ===")
    print(f"  train: MAE {tr_mae:.3f}  HS/LS {tr_acc:.1%}")
    print(f"  test:  MAE {te_mae:.3f}  HS/LS {te_acc:.1%}  "
          f"(predict-0 baseline {base:.3f})")
    # current shipped defaults evaluated on test, for drift visibility
    cur = (0.51, 1.0, 2.0, 2.5)
    cu_mae, cu_acc, _ = _eval(test, cur)
    print(f"  shipped defaults {cur}: test MAE {cu_mae:.3f}  HS/LS {cu_acc:.1%}")

    # per-element diagnostic on TEST at best knobs
    _set_knobs(*best)
    by_el = {}
    for s in test:
        pred = (cfa.site_cf_features(s["species"], s["nbrs"],
                                     default_oxidation=s["oxi"])["cf_unpaired"]
                if s["nbrs"] else 0.0)
        by_el.setdefault(s["el"], []).append((pred, s["m"]))
    print(f"\n=== per-element on TEST at best knobs ===")
    print(f"{'el':>3} {'n':>6} {'MAE':>6} {'<pred>':>7} {'<|m|>':>7} {'corr':>6}")
    for el, pairs in sorted(by_el.items(), key=lambda kv: -len(kv[1])):
        p = np.array(pairs)
        if len(p) < 30:
            continue
        corr = np.corrcoef(p[:, 0], p[:, 1])[0, 1] if len(p) > 2 else float("nan")
        print(f"{el:>3} {len(p):>6} {np.abs(p[:, 0] - p[:, 1]).mean():6.2f} "
              f"{p[:, 0].mean():7.2f} {p[:, 1].mean():7.2f} {corr:6.2f}")
    _set_knobs(0.51, 1.0, 2.0, 2.5)  # restore shipped defaults


if __name__ == "__main__":
    main()
