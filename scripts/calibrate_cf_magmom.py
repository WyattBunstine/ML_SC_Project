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
  3. grid-sweeps (E_SIGMA_SCALE, BETA_NEPHELAUXETIC, K_SD) over the cache,
     scoring MAE(n_unpaired_pred, |m_DFT|) plus a spin-STATE agreement rate on
     the discriminative d4-d7 sites (HS/LS separated by |m|: LS < 1.5 < HS);
  4. reports the surface, the best knobs, and per-element diagnostics.

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


def stream_sample(json_path, n_materials, seed=0):
    """First magmom-carrying frame of every ~k-th material until n collected.
    Deterministic thinning (hash of mp_id) rather than reservoir sampling so
    the same sample re-emerges on re-runs regardless of n."""
    import ijson
    from pymatgen.core import Structure
    rng = random.Random(seed)
    keep_p = 0.25  # thin the ~146k materials; magmom coverage ~14% does the rest
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
    """Worker: structure dict + |m| -> per-TM-site AOM inputs (or None)."""
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
        nbrs[s].append({"vec": v, "weight": float(e["ecn_weight_source"]),
                        "ligand": nodes[t]["element"], "chi": nodes[t]["chi_pauling"]})
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


def gather_sites(json_path, n_materials, cache_path, workers=None):
    """Two-phase: stream raw records (IO-bound), then build in parallel."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    from pymatgen.core import Structure  # noqa: F401  (import check before fork)
    recs = []
    for mp_id, st, mabs in stream_sample(json_path, n_materials):
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


def score(sites, e_scale, beta, k_sd):
    cfa.E_SIGMA_SCALE, cfa.BETA_NEPHELAUXETIC, cfa.K_SD = e_scale, beta, k_sd
    err, n = 0.0, 0
    st_ok = st_tot = 0
    for s in sites:
        f = cfa.site_cf_features(s["species"], s["nbrs"], default_oxidation=s["oxi"])
        pred = f["cf_unpaired"]
        err += abs(pred - s["m"])
        n += 1
        nd = round(s["n_d"])
        if 4 <= nd <= 7:  # HS/LS discriminative counts
            hs_true = s["m"] > 1.5
            # HS prediction: more than half the max-pairing unpaired count
            hs_pred = pred > 1.5
            st_tot += 1
            st_ok += int(hs_true == hs_pred)
    return err / max(n, 1), (st_ok / st_tot if st_tot else float("nan")), n, st_tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="database/datafiles/MPtrj/MPtrj_2022.9_full.json")
    ap.add_argument("--n-materials", type=int, default=1500)
    ap.add_argument("--cache", default="model_data/cf_calib/sites.pickle")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)

    blob = gather_sites(args.json, args.n_materials, args.cache)
    sites = blob["sites"]
    print(f"\ncalibration pool: {len(sites)} TM sites from {blob['n_struct']} structures "
          f"({blob['n_fail']} build failures)")
    m = np.array([s["m"] for s in sites])
    print(f"|magmom| distribution: {np.percentile(m, [5, 25, 50, 75, 95]).round(2)} "
          f"(frac >1.5: {(m > 1.5).mean():.2f})")

    grid_e = [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
    grid_b = [0.45, 0.60, 0.75, 0.90]
    grid_k = [0.0, 1.0, 2.0]
    rows = []
    for k in grid_k:
        for b in grid_b:
            for e in grid_e:
                mae, acc, n, n_st = score(sites, e, b, k)
                rows.append((mae, acc, e, b, k))
    rows.sort()
    print(f"\n=== knob sweep (MAE vs |m_DFT| over {n} sites; "
          f"HS/LS accuracy over {n_st} d4-d7 sites) ===")
    print(f"{'MAE':>7} {'HS/LS':>6} {'e_scale':>8} {'beta':>5} {'k_sd':>5}")
    for mae, acc, e, b, k in rows[:10]:
        print(f"{mae:7.3f} {acc:6.1%} {e:8.2f} {b:5.2f} {k:5.1f}")
    print("   ...")
    for mae, acc, e, b, k in rows[-3:]:
        print(f"{mae:7.3f} {acc:6.1%} {e:8.2f} {b:5.2f} {k:5.1f}")

    # per-element diagnostic at the best point
    mae, acc, e, b, k = rows[0]
    cfa.E_SIGMA_SCALE, cfa.BETA_NEPHELAUXETIC, cfa.K_SD = e, b, k
    by_el = {}
    for s in sites:
        f = cfa.site_cf_features(s["species"], s["nbrs"], default_oxidation=s["oxi"])
        by_el.setdefault(s["el"], []).append((f["cf_unpaired"], s["m"]))
    print(f"\n=== per-element at best (e={e}, beta={b}, k_sd={k}) ===")
    print(f"{'el':>3} {'n':>6} {'MAE':>6} {'<pred>':>7} {'<|m|>':>7}")
    for el, pairs in sorted(by_el.items(), key=lambda kv: -len(kv[1])):
        p = np.array(pairs)
        if len(p) < 20:
            continue
        print(f"{el:>3} {len(p):>6} {np.abs(p[:, 0] - p[:, 1]).mean():6.2f} "
              f"{p[:, 0].mean():7.2f} {p[:, 1].mean():7.2f}")
    # restore defaults
    cfa.E_SIGMA_SCALE, cfa.BETA_NEPHELAUXETIC, cfa.K_SD = 1.0, 0.75, 1.0


if __name__ == "__main__":
    main()
