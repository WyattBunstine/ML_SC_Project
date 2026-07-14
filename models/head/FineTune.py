"""End-to-end fine-tuning of the GPS encoder on T_c (the unfreeze escalation).

The frozen-probe protocol hit a ~5.7K composition ceiling: the encoder carries T_c
signal orthogonal to composition (combo probe 5.99 < raw 7.06) but a frozen head can't
exploit it — capacity to read the 128-d embedding overfits 4k labels, budgeting it away
discards the signal. Fine-tuning attacks that directly: gradients reshape the encoder so
a SMALL DeepSets head can read the T_c-relevant structure.

Guardrails against overfitting 1.5M encoder params on ~4k labels:
  - head WARMUP first (encoder frozen) — never hit a trained encoder with a random head;
  - then unfreeze only the TOP `ft_unfreeze_blocks` GPS block(s) (~332k each), the rest stays frozen;
  - tiny encoder LR (`ft_encoder_lr`) + weight decay, vs the head's normal LR;
  - strong early-stop on val, seed-ensembled test like the frozen head.

Reuses the embed-gps machinery (load_cif_dataset + collate_pool_geom + model.encode) but
in a trainable minibatch loop. Static descriptors + targets + splits come from HeadData so
the split (parent-grouped, leak-free) and the feature set match the frozen runs exactly.
Triggered by `finetune: true` in a head config; dispatched from HeadMain.run.
"""

import copy
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from models.head.HeadData import (assemble, chemsys_groups, family_mae_report,
                                  make_splits, parent_composition_groups)
from models.head.HeadModel import TcHead


def _encoder(checkpoint_path, index_path, device):
    """Rebuild + load the pretrained encoder and the matching graph dataset (same
    feature flags the encoder was trained with), exactly as embed_gps does."""
    for _p in (os.path.join("models", "common"), os.path.join("models", "GPSTransformer")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from data import load_cif_dataset, collate_pool_geom
    from model import GPSCrystalNet
    ckpt = torch.load(checkpoint_path, map_location=device)
    a = ckpt.get("args", {})
    ds = load_cif_dataset(
        index_path, target_column=None, build_angle_bias=True,
        max_num_nbr=a.get("max_num_nbr", 14), max_num_poly_nbr=a.get("max_num_poly_nbr", 16),
        use_poly_edges=a.get("use_poly_edges", True), use_bond_angles=a.get("use_bond_angles", False),
        use_rich_node_features=a.get("use_rich_node_features", False),
        use_valence_features=a.get("use_valence_features", False),
        use_dihedrals=a.get("use_dihedrals", False))
    sa, sn, _, sp, _, _ = ds[0][0][:6]
    model = GPSCrystalNet.from_args(a, (sa.shape[-1], sn.shape[-1], sp.shape[-1]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ds, collate_pool_geom


def run(cfg, out_dir, device):
    from torch.utils.data import DataLoader, Subset

    bs = cfg.get("ft_batch_size", 64)
    pool_dim = cfg["pool_dim"]
    # Loss/early-stop objective in z (= standardized log1p) space: "l1" (default) or
    # "msle" (squared log error -> MSE in z-space, which equals MSLE up to a constant,
    # matching the 3DSC XGBoost objective).
    loss_type = cfg.get("loss", "l1")
    # ---- static tables (descriptors, tc, family) + the leak-free parent-grouped split ----
    # pooling="meanmax" just so assemble doesn't pack the per-atom CSR we don't need here;
    # we only consume phys / tc / family / ids, and the encoder supplies embeddings live.
    data = assemble(cfg["index_path"], cfg["embed_dir"], cfg["descriptors"],
                    cfg["metadata_csv"], pooling="meanmax",
                    aux_meanmax_dir=cfg.get("aux_meanmax_dir"), require_embed=False)
    # Split grouping: "parent" (default, doped variants of one MP parent together),
    # "chemsys" (whole chemical systems — the 3DSC paper's stricter protocol), or
    # "parent_comp" (rounded-cation parent formula — provenance-independent, required
    # once the set mixes MP- and ICSD-parented rows so families don't leak across folds).
    sg = cfg.get("split_group")
    grp = (chemsys_groups(data["ids"]) if sg == "chemsys"
           else parent_composition_groups(data["ids"]) if sg == "parent_comp"
           else None)
    split = make_splits(data, cfg["split_seed"], cfg["val_frac"], cfg["test_frac"], groups=grp)
    ids = list(data["ids"])
    # Optional zero-shot family hold-out: force the listed ids into TEST (and out of
    # train/val) so the model gets ZERO exposure to that family — e.g. the oxide
    # nickelates, to test whether cuprate-learned physics transfers Cu->Ni.
    if cfg.get("holdout_ids_csv"):
        import pandas as _pd
        _ho = set(_pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str))
        _n = sum(1 for r, i in enumerate(ids) if i in _ho and (split.__setitem__(r, "test") or True))
        print(f"[ft] holdout: {_n} ids forced into test (zero train/val exposure)")
    id2row = {i: r for r, i in enumerate(ids)}
    id2split = {i: s for i, s in zip(ids, split)}
    is_sc = data["label"] == 1
    phys_np = data["phys"].astype(np.float32)

    # ---- encoder + graph dataset; align dataset order to the static tables by cif id ----
    model, ds, collate = _encoder(cfg["checkpoint"], cfg["index_path"], device)
    enc_dim = model.atom_fea_len
    # Loader positions MUST be computed in the DATASET's OWN id order: CIFDataV4
    # seed-shuffles its rows at load (build_data_rows), so index-pickle order does
    # not match dataset positions. Mapping split labels through the INDEX order
    # scrambled every fold's actual membership — parent grouping and the nickelate
    # holdout were silently voided (caught 2026-07-14: 30/43 forced-test ids were
    # missing from predictions; test overlap with the intended fold was chance-level).
    ds_ids = [rec[0] for rec in ds.data]
    assert len(ds_ids) == len(id2split) and set(ds_ids) == set(id2split), \
        "dataset/static-table id mismatch"

    def split_indices(s):  # dataset indices that are SC and in split s
        return [i for i, cid in enumerate(ds_ids)
                if id2split.get(cid) == s and is_sc[id2row[cid]]]
    # CIFDataV4 rebuilds each structure's graph on access, so the encode loop is
    # CPU-data-loading-bound; parallel workers overlap graph-building with GPU compute.
    nw = cfg.get("ft_num_workers", 4)
    nw_map = {"train": nw, "val": max(1, nw // 2), "test": 0}

    def mk_loader(s):
        w = nw_map[s]
        return DataLoader(Subset(ds, split_indices(s)), batch_size=bs,
                          shuffle=(s == "train"), collate_fn=collate, num_workers=w,
                          persistent_workers=(w > 0), pin_memory=(str(device) != "cpu"))
    loaders = {s: mk_loader(s) for s in ("train", "val", "test")}
    # sanity: the first train batch's ids must resolve in our static map (order/id alignment)
    _b = next(iter(loaders["train"]))
    assert all(c in id2row for c in _b[3]), "cif-id alignment broken"

    phys_t = torch.as_tensor(phys_np, device=device)
    tc_t = torch.as_tensor(data["tc"], device=device).float()

    def batch_static(cif_ids):
        rows = [id2row[c] for c in cif_ids]
        return phys_t[rows], rows

    def forward_batch(head, inp, cif_ids, grad_encoder):
        inp = tuple(x.to(device) if torch.is_tensor(x) else x for x in inp)
        seg = inp[6].to(device).long()
        if grad_encoder:
            h = model.encode(*inp)
        else:
            with torch.no_grad():
                h = model.encode(*inp)
        phys_b, rows = batch_static(cif_ids)
        _logits, z = head(h, phys_b, seg=seg, n=len(cif_ids))
        return z, rows

    # target / phys normalizers fit ONCE on SC-train (shared across seeds & phases)
    tr_rows = [id2row[c] for c in ds_ids if id2split.get(c) == "train" and is_sc[id2row[c]]]
    tc_tr = tc_t[tr_rows].cpu()

    def make_head(seed):
        torch.manual_seed(seed)
        head = TcHead(enc_dim, phys_t.shape[1], cfg["pca_k"], cfg["hidden"],
                      cfg["dropout"], pooling="deepsets", pool_dim=pool_dim).to(device)
        head.fit_target(tc_tr)
        head.phys_std.fit(phys_t[tr_rows].cpu()); head.to(device)
        return head

    y_z_all = None  # filled per-head (depends on its target norm, but norm is shared → constant)

    def epoch_pass(head, train, opt, y_z):
        head.train()
        # frozen encoder stays in eval (deterministic features); only the fine-tune
        # phase puts the unfrozen blocks in train mode for their own dropout/regularization.
        model.train() if (train and not head._frozen_enc) else model.eval()
        tot = 0.0
        for inp, _t, _l, cif_ids in loaders["train"]:
            opt.zero_grad()
            z, rows = forward_batch(head, inp, cif_ids, grad_encoder=train and not head._frozen_enc)
            loss = F.mse_loss(z, y_z[rows]) if loss_type == "msle" else F.l1_loss(z, y_z[rows])
            loss.backward(); opt.step(); tot += loss.item()
        return tot

    @torch.no_grad()
    def val_z_mae(head, y_z):
        head.eval(); model.eval()
        num = den = 0.0
        for inp, _t, _l, cif_ids in loaders["val"]:
            z, rows = forward_batch(head, inp, cif_ids, grad_encoder=False)
            d = z - y_z[rows]
            num += (d * d).sum().item() if loss_type == "msle" else d.abs().sum().item()
            den += len(rows)
        return num / max(den, 1)

    def fit_phase(head, params, lr, wd, epochs, patience, y_z, tag, log):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        best, best_state, bad, best_ep = np.inf, None, 0, -1
        for ep in range(epochs):
            tr = epoch_pass(head, True, opt, y_z)
            v = val_z_mae(head, y_z)
            if v < best - 1e-6:
                best, bad, best_ep = v, 0, ep
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(head.state_dict()))
            else:
                bad += 1
            log.append({"phase": tag, "epoch": ep, "train_loss": tr, "val_z_mae": v})
            if ep % 20 == 0 or bad >= patience:
                print(f"[ft] {tag} ep{ep:3d} val_z_mae {v:.4f} (best {best:.4f}@{best_ep})", flush=True)
            if bad >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state[0]); head.load_state_dict(best_state[1])
        print(f"[ft] {tag} DONE: best val_z_mae {best:.4f} @ep{best_ep} ({ep + 1} epochs)", flush=True)
        return best

    seed_preds, seed_logs, te_ids = [], [], None
    base_state = copy.deepcopy(model.state_dict())   # restore the pretrained encoder per seed
    for seed in range(cfg.get("n_seeds", 3)):
        model.load_state_dict(base_state)
        head = make_head(seed)
        y_z = head.target_to_z(tc_t)            # tc_t and head buffers both on `device`
        log = []

        # ---- phase A: head warmup (encoder frozen) ----
        for p in model.parameters():
            p.requires_grad = False
        head._frozen_enc = True
        fit_phase(head, head.parameters(), cfg["tc_lr"], cfg["tc_weight_decay"],
                  cfg.get("ft_warmup_epochs", 60), cfg.get("ft_patience", 30), y_z,
                  f"s{seed}-warmup", log)

        # ---- phase B: unfreeze the top blocks, fine-tune at a tiny encoder LR ----
        k = cfg.get("ft_unfreeze_blocks", 1)
        enc_params = []
        for blk in list(model.blocks)[-k:]:
            for p in blk.parameters():
                p.requires_grad = True
                enc_params.append(p)
        head._frozen_enc = False
        val_z = fit_phase(
            head, [{"params": head.parameters(), "lr": cfg["tc_lr"]},
                   {"params": enc_params, "lr": cfg.get("ft_encoder_lr", 1e-5),
                    "weight_decay": cfg.get("ft_encoder_wd", 1e-4)}],
            cfg["tc_lr"], 0.0, cfg.get("ft_epochs", 200), cfg.get("ft_patience", 30),
            y_z, f"s{seed}-finetune", log)

        # ---- test predictions for this seed ----
        model.eval(); head.eval()
        zs, bids = [], []
        with torch.no_grad():
            for inp, _t, _l, cif_ids in loaders["test"]:
                z, _rows = forward_batch(head, inp, cif_ids, grad_encoder=False)
                zs.append(z); bids += list(cif_ids)
        k_pred = head.z_to_kelvin(torch.cat(zs)).cpu().numpy()
        seed_preds.append((bids, k_pred))
        # per-seed single-model test MAE (the ensemble is the headline; this shows spread)
        seed_true = np.array([data["tc"][id2row[c]] for c in bids])
        seed_logs.append({"seed": seed, "val_z_mae": val_z,
                          "test_mae_K": float(np.abs(seed_true - k_pred).mean()),
                          "unfrozen_blocks": k})
        if seed == 0:
            with open(os.path.join(out_dir, "ft_log_seed0.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
            te_ids = bids

    # ---- ensemble in Kelvin over the shared test ids ----
    order = {c: i for i, c in enumerate(te_ids)}
    acc = np.zeros(len(te_ids))
    for bids, k_pred in seed_preds:
        for c, p in zip(bids, k_pred):
            acc[order[c]] += p
    head_K = acc / len(seed_preds)
    rows = [id2row[c] for c in te_ids]
    tc_te = data["tc"][rows]; fam_te = data["family"][rows]; grp_te = data["group"][rows]
    # Global MSLE / RMSE on test (MSLE = the 3DSC paper's metric; head_K already clamped >=0).
    _t = np.asarray(tc_te, float); _p = np.maximum(np.asarray(head_K, float), 0.0)
    msle = float(np.mean((np.log1p(_t) - np.log1p(_p)) ** 2))
    rmse = float(np.sqrt(np.mean((_t - _p) ** 2)))
    metrics = {"finetune": True, "n_seeds": len(seed_preds), "seeds": seed_logs,
               "split_group": cfg.get("split_group", "parent"), "loss": loss_type,
               "msle": msle, "rmse_K": rmse,
               "unfrozen_blocks": cfg.get("ft_unfreeze_blocks", 1),
               "encoder_params_unfrozen": sum(p.numel() for blk in list(model.blocks)[-cfg.get("ft_unfreeze_blocks", 1):] for p in blk.parameters()),
               "head": family_mae_report(tc_te, head_K, fam_te, grp_te)}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(out_dir, "predictions.csv"), "w") as f:
        f.write("id,family,group,tc_true_K,tc_head_K\n")
        for i, c in enumerate(te_ids):
            f.write(f"{c},{fam_te[i]},{grp_te[i]},{tc_te[i]:.3f},{head_K[i]:.3f}\n")
    print(f"[finetune] {len(seed_preds)}-seed ensemble: MAE {metrics['head']['overall']['mae_K']:.2f} K | "
          f"RMSE {rmse:.2f} | MSLE {msle:.3f} | cuprate MAE {metrics['head'].get('family/Cuprate',{}).get('mae_K',float('nan')):.2f} "
          f"[split={metrics['split_group']}, loss={loss_type}, top {cfg.get('ft_unfreeze_blocks',1)} block(s)]")
    print(f"run dir: {out_dir}")
    return metrics
