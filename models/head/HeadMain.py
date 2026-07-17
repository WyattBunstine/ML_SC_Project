"""Trainer for the T_c head on a frozen, pluggable encoder.

Stages per run (config-driven JSON, like the other trainers):

1. **Ridge linear probe** — closed-form on standardized [pooled-enc || phys],
   alpha chosen on val. ~560 effective params. Always runs: this is the
   standard representation-quality metric, reported per family, and the
   baseline every learned head must beat.
2. **Optional trunk pretraining** on the SC/non-SC classification task
   (~61k labels) — weighted cross-entropy, early stop on val AUC. The trunk
   arrives at T_c regression knowing "what distinguishes a superconductor"
   from 10x the labels.
3. **T_c regression** — L1 in z-space (z = znorm(log1p K)), early stop on val
   MAE, over `n_seeds` seeds; the reported head numbers are the seed-ensemble
   (mean z) on the shared test split. If the trunk was pretrained, trunk
   params fine-tune at `lr * trunk_lr_factor`.

Outputs under model_data/<date>/<name>_<timestamp>/: config.json, metadata.json,
splits.csv, metrics.json (probe + head, overall/per-family/per-group MAE in
Kelvin), predictions.csv, seed-0 checkpoint, epoch logs.

Usage: python -m CNN.head.HeadMain configs/head/head_mace_3dsc.json
"""

import copy
import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from models.head.HeadData import assemble, family_mae_report, make_splits
from models.head.HeadModel import Standardizer, TcHead

DEFAULTS = {
    "split_seed": 123, "val_frac": 0.1, "test_frac": 0.2,
    "pca_k": 64, "hidden": 64, "dropout": 0.2,
    # Pooling of the per-atom embedding into one structure vector (see HeadModel):
    # "meanmax" (parameter-free, PCA on [mean||max]), "deepsets", or "attention".
    # pool_dim sizes the learned pools (out width = 2*pool_dim); ignored by meanmax.
    "pooling": "meanmax", "pool_dim": 32,
    "class_pretrain": True, "class_epochs": 300, "class_patience": 30,
    "class_lr": 3e-3, "class_weight_decay": 1e-4,
    "tc_epochs": 2000, "tc_patience": 100, "tc_lr": 3e-3,
    "tc_weight_decay": 1e-4, "trunk_lr_factor": 0.1,
    "n_seeds": 5, "ridge_alphas": [0.1, 1.0, 10.0, 100.0, 1000.0],
    "device": "cpu",
    # Optional transfer-reproducibility fields: with a frozen GPS encoder `checkpoint`
    # set, run() auto-builds the embeddings (from `embed_source`, defaulting to
    # index_path) and the `descriptors` table if they're missing — so ONE config + one
    # `train-head` reproduces embed-gps + descriptors + head. Absent (e.g. the MACE
    # configs, which point embed_dir at a prebuilt dir) -> the artifacts must pre-exist.
    "checkpoint": None, "embed_source": None,
    # Complementarity arm: a second per-atom embedding dir whose [mean||max] is
    # folded into the descriptor vector (e.g. raw mean||max beside a learned gps
    # DeepSets pool — does the encoder add signal orthogonal to composition?).
    "aux_meanmax_dir": None,
    # Fine-tune escalation (models/head/FineTune.py): unfreeze the encoder's top blocks
    # and train end-to-end on T_c after a head warmup. Defaults live in FineTune.run.
    "finetune": False,
}


def _prepare_transfer_inputs(cfg, device):
    """Make the embed_dir + descriptors a head config needs, if absent — so the transfer
    is reproducible from the config alone. Both are cached/resumable: an embed_dir with
    .npy or an existing descriptors pickle is reused untouched."""
    import glob
    embed_dir = cfg["embed_dir"]
    # FineTune encodes graphs LIVE from the checkpoint (assemble require_embed=False) and
    # never reads the precomputed embed_dir, so the (expensive) embedding pass — and its
    # embed_source pack, which need not exist on a compute node — is pure waste for a
    # finetune run. Only the descriptor table below is still needed. (The frozen-probe
    # path DOES consume embed_dir, so it still gets built there.)
    build_embeddings = not cfg.get("finetune")
    # `encoder`: "gps" = the learned encoder embedding (needs a checkpoint); "raw" = the
    # per-atom RAW node features the encoder ingests (ablation: does the encoder add
    # anything?). Defaults to "gps" when a checkpoint is set. Both arms share descriptors
    # so gps-vs-raw isolates the encoder's contribution.
    encoder = cfg.get("encoder") or ("gps" if cfg.get("checkpoint") else None)
    have_emb = os.path.isdir(embed_dir) and glob.glob(os.path.join(embed_dir, "*.npy"))
    if build_embeddings and encoder in ("gps", "raw") and not have_emb:
        # The GPS encoder + data layer live under models/common + models/GPSTransformer;
        # put them on the path the same way the embed-gps CLI does before importing.
        import sys
        for _p in (os.path.join("models", "common"), os.path.join("models", "GPSTransformer")):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        source = cfg.get("embed_source") or cfg["index_path"]   # pack (fast) or the index
        # The raw arm matches the encoder's feature space (rich/dihedral flags) for fairness.
        feat = (torch.load(cfg["checkpoint"], map_location="cpu").get("args", {})
                if cfg.get("checkpoint") else {})
        if encoder == "gps":
            from models.head.embed_gps import embed_index
            print(f"[transfer] GPS-embedding {source} with "
                  f"{os.path.basename(cfg['checkpoint'])} -> {embed_dir}")
            embed_index(cfg["checkpoint"], source, embed_dir, device=device)
        else:  # raw
            from models.head.embed_gps import embed_raw
            print(f"[transfer] RAW node-feature embedding {source} -> {embed_dir}")
            embed_raw(source, embed_dir, feature_args=feat, device=device)
    aux = cfg.get("aux_meanmax_dir")
    if build_embeddings and aux and not (os.path.isdir(aux) and glob.glob(os.path.join(aux, "*.npy"))):
        import sys
        for _p in (os.path.join("models", "common"), os.path.join("models", "GPSTransformer")):
            if _p not in sys.path:
                sys.path.insert(0, _p)
        from models.head.embed_gps import embed_raw
        source = cfg.get("embed_source") or cfg["index_path"]
        feat = (torch.load(cfg["checkpoint"], map_location="cpu").get("args", {})
                if cfg.get("checkpoint") else {})
        print(f"[transfer] building aux RAW mean||max embeddings {source} -> {aux}")
        embed_raw(source, aux, feature_args=feat, device=device)
    if not os.path.exists(cfg["descriptors"]):
        from models.head.descriptors import build_descriptor_table
        print(f"[transfer] building descriptors from {cfg['index_path']} -> {cfg['descriptors']}")
        build_descriptor_table(cfg["index_path"], cfg["descriptors"])


def _to_t(x, device):
    return torch.as_tensor(x, device=device)


def ridge_probe(Xtr, ytr, Xval, yval, Xte, alphas):
    """Closed-form ridge over standardized features; alpha picked on val MAE.

    Returns test predictions (z-space) and the chosen alpha. Bias handled by
    centering y; features are already standardized.
    """
    d = Xtr.shape[1]
    ymean = ytr.mean()
    best = (None, None, np.inf)
    gram = Xtr.T @ Xtr
    xty = Xtr.T @ (ytr - ymean)
    for alpha in alphas:
        w = torch.linalg.solve(gram + alpha * torch.eye(d, dtype=Xtr.dtype), xty)
        val_mae = (Xval @ w + ymean - yval).abs().mean().item()
        if val_mae < best[2]:
            best = (w, alpha, val_mae)
    w, alpha, _ = best
    # Clamp to the train target range: an unbounded linear extrapolation in
    # z-space explodes through the expm1 inverse (observed: 12,544 K).
    return (Xte @ w + ymean).clamp(ytr.min(), ytr.max()), alpha


def train_stage(model, params, batches, loss_fn, val_fn, epochs, patience, lr,
                weight_decay, log_path=None, log_extra=None):
    """Generic full-batch-list training loop with val early stopping.

    `params` is a list of (param_group_dict) for the optimizer; `batches` is a
    callable returning an iterable of (inputs..., target) tuples per epoch;
    `val_fn` returns the validation metric (lower is better) for early stop.
    Restores the best state dict before returning (best_metric, best_epoch).
    """
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    best_metric, best_state, best_epoch, bad = np.inf, None, -1, 0
    log_rows = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in batches():
            opt.zero_grad()
            loss = loss_fn(model, batch)
            loss.backward()
            opt.step()
            total += loss.item()
        model.eval()
        with torch.no_grad():
            metric = val_fn(model)
        improved = metric < best_metric - 1e-6
        if improved:
            best_metric, best_epoch, bad = metric, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        log_rows.append({"epoch": epoch, "train_loss": total, "val_metric": metric,
                         "is_best": int(improved)})
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if log_path:
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(log_rows)
    return best_metric, best_epoch


def run(config_path):
    with open(config_path) as f:
        cfg = {**DEFAULTS, **json.load(f)}
    device = cfg["device"] if (cfg["device"] != "cuda" or torch.cuda.is_available()) else "cpu"

    run_name = cfg.get("name", "head")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("model_data", datetime.now().strftime("%Y-%m-%d"),
                           f"{run_name}_{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    # ---------- reproduce the frozen-encoder artifacts from the config (no-op if cached) ----
    _prepare_transfer_inputs(cfg, device)

    # ---------- fine-tune escalation: encoder in the loop, gradients flow into it ----------
    if cfg.get("finetune"):
        from models.head.FineTune import run as ft_run
        return ft_run(cfg, out_dir, device)

    # ---------------- data ----------------
    pooling = cfg["pooling"]
    learned_pool = pooling != "meanmax"
    data = assemble(cfg["index_path"], cfg["embed_dir"], cfg["descriptors"],
                    cfg["metadata_csv"], pooling=pooling,
                    aux_meanmax_dir=cfg.get("aux_meanmax_dir"))
    split = make_splits(data, cfg["split_seed"], cfg["val_frac"], cfg["test_frac"])
    pd_rows = zip(data["ids"], split, data["label"], data["family"])
    with open(os.path.join(out_dir, "splits.csv"), "w") as f:
        f.write("id,split,label,family\n")
        f.writelines(f"{i},{s},{l},{fam}\n" for i, s, l, fam in pd_rows)

    enc = _to_t(data["enc"], device).float()
    phys = _to_t(data["phys"], device).float()
    tc = _to_t(data["tc"], device).float()
    label = _to_t(data["label"], device)
    is_sc = label == 1
    sc_mask = {s: _to_t((split == s) & is_sc.cpu().numpy(), device) for s in
               ("train", "val", "test")}
    all_mask = {s: _to_t(split == s, device) for s in ("train", "val", "test")}

    # ----- learned-pool inputs: raw per-atom embeddings, gathered per split -----
    # gather() picks the atoms of a structure subset (boolean mask over structures)
    # and renumbers their segment ids 0..n_sub-1 so the pools aggregate per structure.
    # phys[mask] is row-aligned with this gather (both ascending by structure index).
    atom_emb = _to_t(data["atom_emb"], device).float() if learned_pool else None
    atom_ptr = data["atom_ptr"] if learned_pool else None

    def gather(mask_t):
        sel = np.where(mask_t.cpu().numpy())[0]
        lengths = atom_ptr[sel + 1] - atom_ptr[sel]
        atom_idx = np.concatenate([np.arange(atom_ptr[s], atom_ptr[s + 1]) for s in sel]) \
            if len(sel) else np.zeros(0, dtype=np.int64)
        seg = np.repeat(np.arange(len(sel)), lengths)
        return (atom_emb[_to_t(atom_idx, device).long()],
                _to_t(seg, device).long(), len(sel))

    sc_atoms = {s: gather(sc_mask[s]) for s in ("train", "val", "test")} if learned_pool else None
    enc_dim_head = data["atom_dim"] if learned_pool else None  # per-atom width for learned pools

    test_np = sc_mask["test"].cpu().numpy()
    fam_te, grp_te = data["family"][test_np], data["group"][test_np]
    tc_te = data["tc"][test_np]
    metrics = {"n_rows": int(len(data["ids"])),
               "n_sc": int(is_sc.sum()), "fresh_params": None}

    # Target normalizer fit once on SC-train (shared by probe and head).
    norm_model = TcHead(enc.shape[1], phys.shape[1], cfg["pca_k"], cfg["hidden"])
    norm_model.fit_target(tc[sc_mask["train"]].cpu())
    y_z = norm_model.target_to_z(tc.cpu()).to(device)

    # ---------------- stage 1: ridge linear probe ----------------
    probe_std = Standardizer(enc.shape[1] + phys.shape[1]).to(device)
    X_all = torch.cat([enc, phys], dim=1)
    probe_std.fit(X_all[sc_mask["train"]])
    Xs = probe_std(X_all).double()
    z_pred_te, alpha = ridge_probe(
        Xs[sc_mask["train"]], y_z[sc_mask["train"]].double(),
        Xs[sc_mask["val"]], y_z[sc_mask["val"]].double(),
        Xs[sc_mask["test"]], cfg["ridge_alphas"])
    probe_K = norm_model.z_to_kelvin(z_pred_te.float().cpu()).numpy()
    metrics["probe"] = {"alpha": alpha,
                        **family_mae_report(tc_te, probe_K, fam_te, grp_te)}
    print(f"[probe] alpha={alpha}  overall test MAE "
          f"{metrics['probe']['overall']['mae_K']:.2f} K")

    # ---------------- stages 2+3: trunk (optional) + T_c head ----------------
    n_class_train = int(all_mask["train"].sum() - sc_mask["train"].sum())
    do_class = bool(cfg["class_pretrain"]) and n_class_train > 0
    if cfg["class_pretrain"] and not do_class:
        print("[class] skipped: no non-SC rows with embeddings yet")

    if learned_pool and do_class:
        raise NotImplementedError(
            "class pretraining with a learned pool needs all_mask atom gathers; "
            "not wired (the SC-only packs make do_class False anyway).")

    def head_fwd(m, key):
        """Forward the head on SC split `key`, routing meanmax vs learned pooling."""
        if learned_pool:
            emb_s, seg_s, n_s = sc_atoms[key]
            return m(emb_s, phys[sc_mask[key]], seg=seg_s, n=n_s)
        return m(enc[sc_mask[key]], phys[sc_mask[key]])

    seed_preds, seed_logs = [], []
    for seed in range(cfg["n_seeds"]):
        torch.manual_seed(seed)
        model = TcHead(enc_dim_head if learned_pool else enc.shape[1], phys.shape[1],
                       cfg["pca_k"], cfg["hidden"], cfg["dropout"],
                       pooling=pooling, pool_dim=cfg["pool_dim"]).to(device)
        metrics["fresh_params"] = model.n_fresh_params()
        model.fit_target(tc[sc_mask["train"]].cpu())
        # PCA + standardizers fit on the widest training pool this seed sees.
        # (Learned pools have no PCA — they train on the raw per-atom embeddings.)
        fit_mask = all_mask["train"] if do_class else sc_mask["train"]
        if not learned_pool:
            model.pca.fit(enc[fit_mask].cpu())
        model.phys_std.fit(phys[fit_mask].cpu())
        model.to(device)

        if do_class:
            counts = torch.bincount(label[all_mask["train"]], minlength=2).float()
            cls_w = (counts.sum() / (2.0 * counts)).to(device)

            def class_loss(m, _):
                logits, _tc = m(enc[all_mask["train"]], phys[all_mask["train"]])
                return F.cross_entropy(logits, label[all_mask["train"]], weight=cls_w)

            def class_val(m):
                from sklearn.metrics import roc_auc_score
                logits, _tc = m(enc[all_mask["val"]], phys[all_mask["val"]])
                p = logits.softmax(1)[:, 1].cpu().numpy()
                return -roc_auc_score(label[all_mask["val"]].cpu().numpy(), p)

            auc, ep = train_stage(
                model, model.parameters(), lambda: [None], class_loss, class_val,
                cfg["class_epochs"], cfg["class_patience"], cfg["class_lr"],
                cfg["class_weight_decay"],
                log_path=os.path.join(out_dir, f"class_log_seed{seed}.csv")
                if seed == 0 else None)
            if seed == 0:
                metrics["class_pretrain"] = {"val_auc": -auc, "best_epoch": ep,
                                             "n_train": int(all_mask["train"].sum())}
                print(f"[class] seed0 val AUC {-auc:.4f} @ep{ep}")

        groups = [{"params": list(model.class_head.parameters())
                   + list(model.tc_head.parameters())
                   + list(model.norm.parameters()), "lr": cfg["tc_lr"]},
                  {"params": list(model.trunk.parameters()),
                   "lr": cfg["tc_lr"] * (cfg["trunk_lr_factor"] if do_class else 1.0)}]

        def tc_loss(m, _):
            _logits, z = head_fwd(m, "train")
            return F.l1_loss(z, y_z[sc_mask["train"]])

        def tc_val(m):
            _logits, z = head_fwd(m, "val")
            return F.l1_loss(z, y_z[sc_mask["val"]]).item()

        val_mae_z, ep = train_stage(
            model, groups, lambda: [None], tc_loss, tc_val,
            cfg["tc_epochs"], cfg["tc_patience"], cfg["tc_lr"],
            cfg["tc_weight_decay"],
            log_path=os.path.join(out_dir, f"tc_log_seed{seed}.csv")
            if seed == 0 else None)
        model.eval()
        with torch.no_grad():
            _logits, z_te = head_fwd(model, "test")
        seed_preds.append(z_te.cpu())
        seed_logs.append({"seed": seed, "val_mae_z": val_mae_z, "best_epoch": ep})
        if seed == 0:
            torch.save({"state_dict": model.state_dict(), "config": cfg},
                       os.path.join(out_dir, "head_seed0.pth.tar"))

    z_ens = torch.stack(seed_preds).mean(0)
    head_K = norm_model.z_to_kelvin(z_ens).numpy()
    metrics["head"] = {"seeds": seed_logs, "class_pretrained": do_class,
                       **family_mae_report(tc_te, head_K, fam_te, grp_te)}
    print(f"[head]  {cfg['n_seeds']}-seed ensemble overall test MAE "
          f"{metrics['head']['overall']['mae_K']:.2f} K "
          f"({metrics['fresh_params']} fresh params)")

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(out_dir, "predictions.csv"), "w") as f:
        f.write("id,family,group,tc_true_K,tc_probe_K,tc_head_K\n")
        ids_te = data["ids"][test_np]
        for i in range(len(ids_te)):
            f.write(f"{ids_te[i]},{fam_te[i]},{grp_te[i]},{tc_te[i]:.3f},"
                    f"{probe_K[i]:.3f},{head_K[i]:.3f}\n")
    print(f"run dir: {out_dir}")
    return metrics


if __name__ == "__main__":
    run(sys.argv[1])
