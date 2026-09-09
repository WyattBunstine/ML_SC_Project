"""Score NEW materials with a saved fine-tune run — no retraining.

    python scripts/predict_tc.py <run_dir> --index <index.pickle> --descriptors <descriptors.pickle>
                                 [--ids-csv ids.csv] [--out preds.csv] [--device cuda|cpu]

<run_dir> is a FineTune run that wrote model_seed*.pt (FineTune.run with
save_models, the default since 2026-09-09) next to config.json. The new
materials need (1) graphs built with the SAME pipeline as the run's own index
(the v45 doped graph builder; an index pickle with id + graph_path columns) and
(2) a descriptors pickle {"names", "table": {id: vector}} in the run's
descriptor layout — the names are checked against the ones saved with the
model. Every seed is rebuilt (base checkpoint named in config.json + the
saved fine-tuned tensors + the saved head) and the seeds are ensembled exactly
as FineTune does (Kelvin mean, or log-space mean for ensemble_space "log").
pred_features arms (P/G/LP/...) are not supported here (they need the
prediction cache); latents-only arms — every fam_/la_series/probe run — are.
If the index carries a tc/value column, an MAE over rows with tc>0 is printed.
"""
import argparse
import copy
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "models", "common"), os.path.join(_ROOT, "models", "GPSTransformer"),
           os.path.join(_ROOT, "models", "head")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from models.head.FineTune import _encoder  # noqa: E402
from models.head.HeadModel import TcHead  # noqa: E402
from data import dataset_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--index", required=True, help="index pickle of the materials to score (id, graph_path)")
    ap.add_argument("--descriptors", required=True, help="descriptors pickle covering those ids")
    ap.add_argument("--ids-csv", default=None, help="optional csv with an 'id' column: score only these")
    ap.add_argument("--out", default=None, help="output csv (default <run_dir>/predictions_new.csv)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=6, help="DataLoader workers (graph JSON parsing is the bottleneck)")
    a = ap.parse_args()
    run = a.run_dir.rstrip("/")
    cfg = json.load(open(os.path.join(run, "config.json")))
    files = sorted(glob.glob(os.path.join(run, "model_seed*.pt")))
    if not files:
        sys.exit(f"{run}: no model_seed*.pt — the run predates model saving; retrain with save_models")
    if cfg.get("pred_features"):
        sys.exit("pred_features arms are not supported by predict_tc.py (prediction-cache features)")
    device = torch.device(a.device)
    # ---- encoder + graph dataset for the NEW index (same feature space as the run) ----
    model, ds, collate = _encoder(cfg.get("checkpoint"), a.index, device, encoder_args=cfg.get("encoder_args"))
    base_state = copy.deepcopy(model.state_dict())
    ids = list(dataset_ids(ds))
    keep = set(pd.read_csv(a.ids_csv)["id"].astype(str)) if a.ids_csv else set(ids)
    sel = [i for i, c in enumerate(ids) if c in keep]
    missing = sorted(keep - set(ids))
    if missing:
        print(f"[predict] {len(missing)} requested ids are not in the index (e.g. {missing[:3]})")
    if not sel:
        sys.exit("nothing to score")
    # ---- descriptors in the run's layout ----
    desc = pd.read_pickle(a.descriptors)
    table, names = desc["table"], list(desc["names"])
    ref = torch.load(files[0], map_location="cpu", weights_only=False)
    if ref.get("desc_names") and list(ref["desc_names"]) != names:
        sys.exit(f"descriptor layout mismatch: run used {len(ref['desc_names'])} names, given {len(names)}")
    nodesc = [ids[i] for i in sel if ids[i] not in table]
    if nodesc:
        sys.exit(f"{len(nodesc)} ids have no descriptor row (e.g. {nodesc[:3]})")
    phys = {ids[i]: torch.as_tensor(np.asarray(table[ids[i]], dtype=np.float32)) for i in sel}
    loader = DataLoader(Subset(ds, sel), batch_size=a.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=a.workers, persistent_workers=(a.workers > 0))
    # ---- every seed instantiated up front, graphs loaded ONCE (the loading is the cost) ----
    seeds = []
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        model.load_state_dict(base_state)
        sd = model.state_dict()
        sd.update({k: v.to(sd[k].device) if k in sd else v for k, v in blob["encoder_state"].items()})
        model.load_state_dict(sd)
        m_ = copy.deepcopy(model).eval()
        head = TcHead(**blob["head_kwargs"]).to(device)
        head.load_state_dict(blob["head_state"]); head.eval()
        seeds.append((blob, m_, head))
        print(f"[predict] seed {blob['seed']} ({blob['unfreeze']}, {len(blob['encoder_state'])} encoder tensors) loaded", flush=True)
    ks = [[] for _ in seeds]; ps = [[] for _ in seeds]; order = []
    n_done = 0
    with torch.no_grad():
        for inp, _t, _l, cif_ids in loader:
            inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in inp)
            seg = inp[6].to(device).long()
            phys_b = torch.stack([phys[c] for c in cif_ids]).to(device)
            for j, (blob, m_, head) in enumerate(seeds):
                h = m_.encode(*inp)
                logits, z = head(h, phys_b, seg=seg, n=len(cif_ids))
                k = head.z_to_kelvin(z)
                pr = torch.softmax(logits, dim=1)
                if blob.get("use_gs"):
                    k = pr[:, 0] * k
                ks[j].append(k.cpu()); ps[j].append(pr.cpu())
            order += list(cif_ids)
            n_done += len(cif_ids)
            if n_done % (a.batch_size * 200) < a.batch_size:
                print(f"[predict] {n_done}/{len(sel)}", flush=True)
    seed_k = [torch.cat(k).numpy() for k in ks]; seed_p = [torch.cat(p_).numpy() for p_ in ps]
    K = np.stack(seed_k)                                   # (seeds, n)
    log_ens = ref.get("ensemble_space") == "log"
    ens = np.expm1(np.log1p(np.maximum(K, 0)).mean(0)) if log_ens else K.mean(0)
    P = np.stack(seed_p).mean(0)
    out = pd.DataFrame({"id": order, "tc_pred_K": np.round(ens, 3), "tc_pred_std_K": np.round(K.std(0), 3)})
    if ref.get("use_gs"):
        out["p_sc"] = np.round(P[:, 0], 4)
    for s_, k in enumerate(K):
        out[f"seed{s_}_K"] = np.round(k, 3)
    idx = pd.read_pickle(a.index)
    tcol = next((c for c in ("tc", "value") if c in idx.columns), None)
    if tcol:
        t = dict(zip(idx["id"].astype(str), idx[tcol]))
        out["tc_true_K"] = [t.get(c, np.nan) for c in order]
        pos = out.tc_true_K > 0
        if pos.any():
            print(f"[predict] rows with tc>0: {int(pos.sum())}, MAE {np.abs(out.tc_true_K[pos] - out.tc_pred_K[pos]).mean():.2f} K")
    dest = a.out or os.path.join(run, "predictions_new.csv")
    out.to_csv(dest, index=False)
    print(f"[predict] {len(out)} materials x {len(files)} seeds -> {dest}")


if __name__ == "__main__":
    main()
