"""End-to-end fine-tuning of the GPS encoder on T_c (the unfreeze escalation).

The frozen-probe protocol hit a ~5.7K composition ceiling: the encoder carries T_c
signal orthogonal to composition (combo probe 5.99 < raw 7.06) but a frozen head can't
exploit it — capacity to read the 128-d embedding overfits 4k labels, budgeting it away
discards the signal. Fine-tuning attacks that directly: gradients reshape the encoder so
a SMALL DeepSets head can read the T_c-relevant structure.

Loss spaces (config "loss", validated at startup; shared with the head HPO
sweep via module-level reg_loss): l1 / msle (z-space) and mse_k / wl1_k
(Kelvin-space); "ensemble_space" log|kelvin picks the seed-ensemble average
domain. Unfreeze modes (ft_unfreeze): blocks:k / norms / embedding / bitfit /
ffn:k / attn:k — norms (3k params) is the transfer-probe default.

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
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from models.head.embed_cache import build_embed_cache, cat_cached
from models.head.HeadData import (assemble, chemsys_groups, family_mae_report,
                                  make_splits, parent_composition_groups)
from models.head.HeadModel import TcHead


def _encoder(checkpoint_path, index_path, device):
    """Rebuild + load the pretrained encoder and the matching graph dataset. The
    feature-flag enumeration lives in ONE place (data.load_cif_dataset_from_args,
    shared with embed_index/embed_raw/smoke) so this dataset is always built in the
    exact feature space the encoder was trained with — no per-consumer drift."""
    for _p in (os.path.join("models", "common"), os.path.join("models", "GPSTransformer")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    from data import load_cif_dataset_from_args, collate_pool_geom
    from model import GPSCrystalNet
    ckpt = torch.load(checkpoint_path, map_location=device)
    a = ckpt.get("args", {})
    ds = load_cif_dataset_from_args(index_path, a)
    sa, sn, _, sp, _, _ = ds[0][0][:6]
    model = GPSCrystalNet.from_args(a, (sa.shape[-1], sn.shape[-1], sp.shape[-1]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device), ds, collate_pool_geom


VALID_LOSSES = ("l1", "msle", "mse_k", "wl1_k", "huber_k")

# huber_k transition point (Kelvin): quadratic below (mse_k-like amplitude
# seeking), linear above (robust to the high-Tc label-noise rows that make
# pure mse_k gradients outlier-dominated). ~ the SC-only MAE scale.
HUBER_BETA_K = 10.0


def reg_loss(z, y_z_sel, tc_sel, loss_type, tc_mean, tc_std, k_var=1.0, k_wmean=1.0):
    """One regression term, per the configured loss space. Shared by the fine-tune
    protocol (run) and the stage-A head HPO sweep so the two can't drift.
      l1     — L1 in z (log1p-standardized): the historical default; small-error
               mass dominates, family-scale misses on rare rows can hide.
      msle   — MSE in z: emphasizes RELATIVE error (a 0.5->2 K miss ~ a 30->120).
      mse_k  — MSE in KELVIN: a 24 K miss costs 576x a 1 K miss — targets the
               don't-care-about-small-errors regime (found: the val09 champion
               under-predicted every >20 K iron-based by ~24 K and L1 never saw
               it). Kelvin map is unclamped below (gradients survive pred<0) but
               the exponent is capped at 30 (~1e13 K, far beyond physics): fp32
               expm1 overflows near 88 and the MSE square near 44, and one inf
               loss NaN-poisons the weights for the rest of the run. Normalized
               by Var(Tc_train) for optimizer-scale sanity.
      wl1_k  — L1 in z weighted by (1+Tc_true): the same emphasis direction,
               robust to high-Tc label noise (the 130 K near-duplicate problem).
      huber_k — SmoothL1 in KELVIN (beta = HUBER_BETA_K): quadratic below beta
               (amplitude-seeking like mse_k), linear above (a 100 K outlier
               costs ~10x a 10 K miss, not 100x). Same exponent cap as mse_k;
               normalized by beta so the tail gradient is ~L1-in-z scale.
    """
    if loss_type == "msle":
        return F.mse_loss(z, y_z_sel)
    if loss_type == "mse_k":
        k = torch.expm1((z * tc_std + tc_mean).clamp(max=30.0))
        return F.mse_loss(k, tc_sel) / k_var
    if loss_type == "wl1_k":
        w = (1.0 + tc_sel) / k_wmean
        return (w * (z - y_z_sel).abs()).mean()
    if loss_type == "huber_k":
        k = torch.expm1((z * tc_std + tc_mean).clamp(max=30.0))
        return F.smooth_l1_loss(k, tc_sel, beta=HUBER_BETA_K) / HUBER_BETA_K
    if loss_type == "l1":
        return F.l1_loss(z, y_z_sel)
    raise ValueError(f"unknown loss {loss_type!r} (use one of {VALID_LOSSES})")


def run(cfg, out_dir, device):
    from torch.utils.data import DataLoader, Subset

    bs = cfg.get("ft_batch_size", 64)
    pool_dim = cfg["pool_dim"]
    # Loss/early-stop objective (see reg_loss for the spaces). Validated HERE so a
    # typo'd config fails at startup instead of silently training the l1 fallback
    # while metrics.json records the intended loss name.
    loss_type = cfg.get("loss", "l1")
    if loss_type not in VALID_LOSSES:
        raise ValueError(f"unknown loss {loss_type!r} (use one of {VALID_LOSSES})")
    ens_space = cfg.get("ensemble_space", "kelvin")
    if ens_space not in ("kelvin", "log"):
        raise ValueError(f"unknown ensemble_space {ens_space!r} (use 'kelvin' or 'log')")
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
    # Optional curated exclusions (label artifacts — see docs/data_curation/):
    # rows dropped ENTIRELY before splitting, never train/val/test. Off by
    # default; adopt per evaluation ROUND so all compared arms share it.
    if cfg.get("exclude_ids_csv"):
        import pandas as _pd
        _ex = set(_pd.read_csv(cfg["exclude_ids_csv"])["id"].astype(str))
        _keep = [r for r, i in enumerate(ids) if i not in _ex]
        _n_drop = len(ids) - len(_keep)
        import numpy as _np
        _keep = _np.asarray(_keep)
        for _key in ("phys", "tc", "label", "family", "group", "gs"):
            if _key in data:
                data[_key] = data[_key][_keep]
        data["ids"] = [ids[r] for r in _keep]
        split = [split[r] for r in _keep]
        ids = list(data["ids"])
        print(f"[ft] exclusions: dropped {_n_drop} curated label-artifact rows")

    # Optional forced-TRAIN ids: family relatives that must supervise the model
    # (e.g. other-dopant La-cuprate variants when only the Sr/Ce series is the
    # held-out question). Applied BEFORE the holdout force so a conflicting id
    # ends up in test — the zero-exposure guarantee always wins.
    if cfg.get("train_ids_csv"):
        import pandas as _pd
        _tr = set(_pd.read_csv(cfg["train_ids_csv"])["id"].astype(str))
        _n = 0
        for r, i in enumerate(ids):
            if i in _tr:
                split[r] = "train"
                _n += 1
        print(f"[ft] train-force: {_n} ids forced into train")
    if cfg.get("holdout_ids_csv"):
        import pandas as _pd
        _ho = set(_pd.read_csv(cfg["holdout_ids_csv"])["id"].astype(str))
        if cfg.get("train_ids_csv"):
            _both = _ho & set(_pd.read_csv(cfg["train_ids_csv"])["id"].astype(str))
            if _both:
                print(f"[ft] WARN: {len(_both)} ids in BOTH train-force and holdout -> held out")
        _n = 0
        for r, i in enumerate(ids):
            if i in _ho:
                split[r] = "test"
                _n += 1
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
    from data import dataset_ids
    ds_ids = dataset_ids(ds)
    # Every id in the static tables must resolve in the dataset (the 2026-07-14
    # positional-split lesson). The dataset MAY carry extra rows — exactly the
    # exclude_ids_csv case: excluded ids stay in the index/dataset but are
    # absent from id2split, so split_indices routes them to no fold.
    assert set(id2split) <= set(ds_ids), "dataset/static-table id mismatch"
    if len(ds_ids) != len(id2split):
        print(f"[ft] dataset carries {len(ds_ids) - len(id2split)} rows outside "
              "the static tables (exclusions) — never loaded")

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
    # (No first-batch spin-up check here: the ds_ids set-equality assert above already
    # guarantees every dataset id resolves in the static tables.)

    phys_t = torch.as_tensor(phys_np, device=device)
    tc_t = torch.as_tensor(data["tc"], device=device).float()
    # ---- ground-state HURDLE mode (phase 2 of the magnetic negatives) ----
    # ground_state_head: the head grows a 4-class ground-state output (SC/FM/AFM/both,
    # labels from data["gs"], -1 = unknown -> masked out of the CE), the REGRESSION
    # trains on tc>0 rows ONLY (negatives inform the classifier, never drag the Tc
    # scale — the fix for the modulation-vs-discrimination trade the tc=0-flood
    # experiment exposed), and inference is the hurdle E[Tc] = P(SC) * Tc_reg.
    use_gs = bool(cfg.get("ground_state_head", False))
    gs_w = float(cfg.get("gs_loss_weight", 1.0))
    gs_t = torch.as_tensor(data["gs"], device=device).long()
    is_pos = tc_t > 0                                   # regression-supervised rows
    # BINARY hurdle (gs_binary + n_gs_classes:2): no mag_order labels needed —
    # every tc=0 row is an observed non-superconductor (3DSC convention), so it
    # supervises class 1 directly instead of being masked. Without this, an index
    # lacking mag_order gives the classifier zero negatives and P(SC) degenerates
    # to ~1 (the hurdle silently becomes plain regression-on-positives).
    if use_gs and bool(cfg.get("gs_binary", False)):
        gs_t = (~is_pos).long()

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
        logits, z = head(h, phys_b, seg=seg, n=len(cif_ids))
        return logits, z, rows

    def _reg_loss(head, z, y_z_sel, tc_sel):
        # Delegates to the shared module-level reg_loss (also used by the head HPO
        # sweep); the head supplies the train-fit z<->Kelvin constants for mse_k.
        return reg_loss(z, y_z_sel, tc_sel, loss_type,
                        head.tc_mean, head.tc_std, k_var, k_wmean)

    def joint_loss(head, logits, z, rows, y_z):
        """Legacy: plain regression over the batch. Hurdle: regression on tc>0 rows +
        masked CE on ground-state-labeled rows; a batch missing one population
        contributes only the other term (never NaN)."""
        rt = torch.as_tensor(rows, device=device)
        if not use_gs:
            return _reg_loss(head, z, y_z[rt], tc_t[rt])
        loss = z.sum() * 0.0
        m_pos = is_pos[rt]
        if m_pos.any():
            loss = loss + _reg_loss(head, z[m_pos], y_z[rt][m_pos], tc_t[rt][m_pos])
        m_gs = gs_t[rt] >= 0
        if m_gs.any():
            loss = loss + gs_w * F.cross_entropy(logits[m_gs], gs_t[rt][m_gs])
        return loss

    # target / phys normalizers fit ONCE on train (shared across seeds & phases).
    # Hurdle mode fits the Tc normalizer on the tc>0 train rows only — the regression
    # never sees the zeros, so its z-space must not be centered by them.
    tr_rows = [id2row[c] for c in ds_ids if id2split.get(c) == "train" and is_sc[id2row[c]]]
    fit_rows = ([r for r in tr_rows if bool(is_pos[r])] if use_gs else tr_rows)
    tc_tr = tc_t[fit_rows].cpu()
    # Kelvin-space loss normalizers (train-fit constants; see _reg_loss)
    k_var = max(float(tc_tr.var()), 1.0)
    k_wmean = max(float((1.0 + tc_tr).mean()), 1.0)

    def make_head(seed):
        torch.manual_seed(seed)
        head = TcHead(enc_dim, phys_t.shape[1], cfg["pca_k"], cfg["hidden"],
                      cfg["dropout"], pooling="deepsets", pool_dim=pool_dim,
                      n_classes=(int(cfg.get("n_gs_classes", 4)) if use_gs else 2),
                      head_arch=cfg.get("head_arch", "concat"),
                      pool_rank=cfg.get("pool_rank"),
                      pool_agg=cfg.get("pool_agg", "meanmax")).to(device)
        head.fit_target(tc_tr)
        head.phys_std.fit(phys_t[tr_rows].cpu()); head.to(device)
        return head

    # ---- phase-A embedding cache: the FROZEN encoder's per-atom h for train+val ----
    # Warmup never updates the encoder (it is reset to base_state each seed and stays
    # frozen + eval throughout phase A), so its per-atom embeddings are bit-identical
    # across every warmup epoch AND every seed. Encode train/val ONCE here and train
    # the head on cached tensors — removes ~(warmup_epochs x n_seeds) redundant full
    # encoder passes + per-access JSON graph rebuilds. Phase B fine-tunes the encoder,
    # so it keeps encoding live through the loaders.
    h_cache = {"train": build_embed_cache(model, loaders["train"], device),
               "val": build_embed_cache(model, loaders["val"], device)}
    train_ids_sc = [ds_ids[i] for i in split_indices("train")]
    val_ids_sc = [ds_ids[i] for i in split_indices("val")]

    def _cached_forward(head, split, bids):
        h, seg = cat_cached(h_cache[split], bids, device)
        phys_b, rows = batch_static(bids)
        logits, z = head(h, phys_b, seg=seg, n=len(bids))
        return logits, z, rows

    def warmup_epoch(head, opt, y_z, rng):
        head.train(); model.eval()
        order = list(train_ids_sc)
        rng.shuffle(order)
        tot = 0.0
        for b0 in range(0, len(order), bs):
            bids = order[b0:b0 + bs]
            opt.zero_grad()
            logits, z, rows = _cached_forward(head, "train", bids)
            loss = joint_loss(head, logits, z, rows, y_z)
            loss.backward(); opt.step(); tot += loss.item()
        return tot

    def _val_metric(head, logits, z, rows, y_z):
        """Early-stop objective on a val batch -> (sum, count), in the SAME loss space
        as training (_reg_loss), so model selection optimizes what the loss optimizes.
        Hurdle: regression term over tc>0 rows + weighted CE over labeled rows."""
        rt = torch.as_tensor(rows, device=device)
        if not use_gs:
            return float(_reg_loss(head, z, y_z[rt], tc_t[rt])) * len(rows), len(rows)
        s, cnt = 0.0, 0
        m_pos = is_pos[rt]
        if m_pos.any():
            n = int(m_pos.sum())
            s += float(_reg_loss(head, z[m_pos], y_z[rt][m_pos], tc_t[rt][m_pos])) * n
            cnt += n
        m_gs = gs_t[rt] >= 0
        if m_gs.any():
            s += gs_w * float(F.cross_entropy(logits[m_gs], gs_t[rt][m_gs],
                                              reduction="sum"))
            cnt += int(m_gs.sum())
        return s, cnt

    @torch.no_grad()
    def warmup_val(head, y_z):
        head.eval()
        num = den = 0.0
        for b0 in range(0, len(val_ids_sc), bs):
            bids = val_ids_sc[b0:b0 + bs]
            logits, z, rows = _cached_forward(head, "val", bids)
            s, c = _val_metric(head, logits, z, rows, y_z)
            num += s; den += c
        return num / max(den, 1)

    def epoch_pass(head, train, opt, y_z):
        head.train()
        # frozen encoder stays in eval (deterministic features); only the fine-tune
        # phase puts the unfrozen blocks in train mode for their own dropout/regularization.
        model.train() if (train and not head._frozen_enc) else model.eval()
        tot = 0.0
        for inp, _t, _l, cif_ids in loaders["train"]:
            opt.zero_grad()
            logits, z, rows = forward_batch(head, inp, cif_ids,
                                            grad_encoder=train and not head._frozen_enc)
            loss = joint_loss(head, logits, z, rows, y_z)
            loss.backward(); opt.step(); tot += loss.item()
        return tot

    @torch.no_grad()
    def val_z_mae(head, y_z):
        head.eval(); model.eval()
        num = den = 0.0
        for inp, _t, _l, cif_ids in loaders["val"]:
            logits, z, rows = forward_batch(head, inp, cif_ids, grad_encoder=False)
            s, c = _val_metric(head, logits, z, rows, y_z)
            num += s; den += c
        return num / max(den, 1)

    def fit_phase(head, params, lr, wd, epochs, patience, y_z, tag, log,
                  epoch_fn=None, val_fn=None):
        # epoch_fn/val_fn: the cached-embedding warmup path (phase A); default = the
        # live-encoding loaders path (phase B, where the encoder is training).
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        best, best_state, bad, best_ep = np.inf, None, 0, -1
        for ep in range(epochs):
            tr = epoch_fn(head, opt, y_z) if epoch_fn else epoch_pass(head, True, opt, y_z)
            v = val_fn(head, y_z) if val_fn else val_z_mae(head, y_z)
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

    seed_preds, seed_probs, seed_logs, te_ids = [], [], [], None
    base_state = copy.deepcopy(model.state_dict())   # restore the pretrained encoder per seed
    for seed in range(cfg.get("n_seeds", 3)):
        model.load_state_dict(base_state)
        head = make_head(seed)
        y_z = head.target_to_z(tc_t)            # tc_t and head buffers both on `device`
        log = []

        # ---- phase A: head warmup (encoder frozen; cached embeddings, no re-encode) ----
        for p in model.parameters():
            p.requires_grad = False
        head._frozen_enc = True
        _rng = random.Random(seed)                     # deterministic warmup batch order
        fit_phase(head, head.parameters(), cfg["tc_lr"], cfg["tc_weight_decay"],
                  cfg.get("ft_warmup_epochs", 60), cfg.get("ft_patience", 30), y_z,
                  f"s{seed}-warmup", log,
                  epoch_fn=lambda h_, o_, y_: warmup_epoch(h_, o_, y_, _rng),
                  val_fn=warmup_val)

        # ---- phase B: unfreeze the selected encoder subset, tiny encoder LR ----
        # ft_unfreeze picks WHERE gradients enter the encoder (params in parens):
        #   "blocks:k"  top-k GPS blocks (default, k=ft_unfreeze_blocks; 333k/block)
        #   "norms"     every LayerNorm (3.1k)  — global scale recalibration
        #   "bitfit"    every bias (12k)        — slightly richer recalibration
        #   "embedding" the input Linear (2.4k) — re-mix input features (e.g. re-weight
        #               the valence dims that physics pretraining weighted for energies)
        #   "ffn:k"     the FFNs of the top-k blocks (132k/block) — computation, not attention
        #   "attn:k"    the shell attentions of the top-k blocks (200k/block)
        mode = cfg.get("ft_unfreeze", f"blocks:{cfg.get('ft_unfreeze_blocks', 1)}")
        kind, _, karg = mode.partition(":")
        k = int(karg) if karg else 1
        if kind == "blocks":
            mods = list(model.blocks)[-k:]
        elif kind == "ffn":
            mods = [m for blk in list(model.blocks)[-k:]
                    for n, m in blk.named_children() if n.startswith("ffn")]
        elif kind == "attn":
            mods = [m for blk in list(model.blocks)[-k:]
                    for n, m in blk.named_children() if n.endswith("_attn")]
        elif kind == "embedding":
            mods = [model.embedding]
        elif kind in ("norms", "bitfit"):
            mods = []                               # selected by parameter name below
        else:
            raise ValueError(f"unknown ft_unfreeze mode {mode!r}")
        enc_params = [p for m_ in mods for p in m_.parameters()]
        if kind == "norms":
            enc_params = [p for n, p in model.named_parameters()
                          if "norm" in n.lower() and "heads." not in n]
        elif kind == "bitfit":
            enc_params = [p for n, p in model.named_parameters()
                          if n.endswith(".bias") and "heads." not in n]
        for p in enc_params:
            p.requires_grad = True
        if seed == 0:
            print(f"[ft] phase-B unfreeze '{mode}': "
                  f"{sum(p.numel() for p in enc_params):,} encoder params", flush=True)
        head._frozen_enc = False
        # The head KEEPS its warmup weight decay in phase B (each param group sets its
        # own; the optimizer-level default would otherwise silently zero it, leaving
        # the head unregularized during exactly the highest-capacity phase).
        val_z = fit_phase(
            head, [{"params": head.parameters(), "lr": cfg["tc_lr"],
                    "weight_decay": cfg["tc_weight_decay"]},
                   {"params": enc_params, "lr": cfg.get("ft_encoder_lr", 1e-5),
                    "weight_decay": cfg.get("ft_encoder_wd", 1e-4)}],
            cfg["tc_lr"], 0.0, cfg.get("ft_epochs", 200), cfg.get("ft_patience", 30),
            y_z, f"s{seed}-finetune", log)

        # ---- test predictions for this seed ----
        model.eval(); head.eval()
        zs, lgs, bids = [], [], []
        with torch.no_grad():
            for inp, _t, _l, cif_ids in loaders["test"]:
                logits, z, _rows = forward_batch(head, inp, cif_ids, grad_encoder=False)
                zs.append(z); lgs.append(logits); bids += list(cif_ids)
        k_reg = head.z_to_kelvin(torch.cat(zs)).cpu().numpy()
        probs = torch.softmax(torch.cat(lgs), dim=1).cpu().numpy()
        if use_gs:
            # hurdle: E[Tc] = P(SC) * Tc_reg — the classifier gates the scale, the
            # regression (trained on SC only) carries it.
            k_pred = probs[:, 0] * k_reg
        else:
            k_pred = k_reg
        seed_preds.append((bids, k_pred))
        seed_probs.append((bids, probs))
        # per-seed single-model test MAE (the ensemble is the headline; this shows
        # spread). Named explicitly: _all_K pools every test row INCLUDING tc=0
        # (flattered by easy zeros on negative-rich pools), _pos_K is SC-only —
        # the one comparable to the headline mae_tc_pos_K.
        seed_true = np.array([data["tc"][id2row[c]] for c in bids])
        seed_pos = seed_true > 0
        seed_logs.append({"seed": seed, "val_z_mae": val_z,
                          "test_mae_all_K": float(np.abs(seed_true - k_pred).mean()),
                          "test_mae_pos_K": (float(np.abs(seed_true[seed_pos]
                                                          - k_pred[seed_pos]).mean())
                                             if seed_pos.any() else None),
                          "unfreeze": mode,
                          "encoder_params_unfrozen": int(sum(p.numel() for p in enc_params))})
        if seed == 0:
            with open(os.path.join(out_dir, "ft_log_seed0.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(log[0].keys())); w.writeheader(); w.writerows(log)
            te_ids = bids

    # ---- ensemble over the shared test ids: Kelvin-mean (default) or log-space ----
    # ensemble_space "log": average seeds in log1p-Kelvin and expm1 back — the right
    # aggregation when the reporting metric is MSLE (arithmetic Kelvin means are
    # biased high in log space; the chemsys/XGBoost comparison is scored on MSLE).
    log_ens = ens_space == "log"
    order = {c: i for i, c in enumerate(te_ids)}
    acc = np.zeros(len(te_ids))
    for bids, k_pred in seed_preds:
        for c, p in zip(bids, k_pred):
            acc[order[c]] += np.log1p(max(p, 0.0)) if log_ens else p
    head_K = np.expm1(acc / len(seed_preds)) if log_ens else acc / len(seed_preds)
    n_cls = seed_probs[0][1].shape[1]
    prob_acc = np.zeros((len(te_ids), n_cls))
    for bids, probs in seed_probs:
        for c, pr in zip(bids, probs):
            prob_acc[order[c]] += pr
    prob_te = prob_acc / len(seed_probs)
    rows = [id2row[c] for c in te_ids]
    tc_te = data["tc"][rows]; fam_te = data["family"][rows]; grp_te = data["group"][rows]
    gs_te = data["gs"][rows]
    # Global MSLE / RMSE on test (MSLE = the 3DSC paper's metric; head_K already clamped >=0).
    _t = np.asarray(tc_te, float); _p = np.maximum(np.asarray(head_K, float), 0.0)
    msle = float(np.mean((np.log1p(_t) - np.log1p(_p)) ** 2))
    rmse = float(np.sqrt(np.mean((_t - _p) ** 2)))
    # SC-only aggregate: on indexes with many tc=0 rows (magnetic negatives, parents)
    # the pooled MAE is flattered by easy zeros (a 77%-zero test pool turned 7.0 K
    # per-superconductor into a 3.05 headline) — report the tc>0 number ALWAYS.
    _pos = _t > 0
    mae_pos = float(np.abs(_t[_pos] - _p[_pos]).mean()) if _pos.any() else None
    metrics = {"finetune": True, "n_seeds": len(seed_preds), "seeds": seed_logs,
               "split_group": cfg.get("split_group", "parent"), "loss": loss_type,
               "msle": msle, "rmse_K": rmse,
               "mae_tc_pos_K": mae_pos,
               "n_test_tc_pos": int(_pos.sum()), "n_test_tc_zero": int((~_pos).sum()),
               "unfreeze": mode,
               "encoder_params_unfrozen": int(sum(p.numel() for p in enc_params)),
               "head": family_mae_report(tc_te, head_K, fam_te, grp_te)}
    if use_gs:
        # ground-state classifier report over the LABELED test rows (gs>=0):
        # per-class precision/recall from the ensemble-averaged probabilities.
        GS_NAMES = ["SC", "FM", "AFM", "both"][:n_cls]
        lab = gs_te >= 0
        y, yhat = gs_te[lab], prob_te[lab].argmax(1)
        cls_report = {"n_labeled_test": int(lab.sum()),
                      "accuracy": float((y == yhat).mean()) if lab.any() else None}
        for k_, nm in enumerate(GS_NAMES):
            tp = int(((yhat == k_) & (y == k_)).sum())
            cls_report[nm] = {"n": int((y == k_).sum()),
                              "precision": round(tp / max(int((yhat == k_).sum()), 1), 3),
                              "recall": round(tp / max(int((y == k_).sum()), 1), 3)}
        metrics["ground_state"] = cls_report
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(out_dir, "predictions.csv"), "w") as f:
        extra = ",p_sc,gs_true,gs_pred" if use_gs else ""
        f.write(f"id,family,group,tc_true_K,tc_head_K{extra}\n")
        for i, c in enumerate(te_ids):
            base_row = f"{c},{fam_te[i]},{grp_te[i]},{tc_te[i]:.3f},{head_K[i]:.3f}"
            if use_gs:
                base_row += f",{prob_te[i, 0]:.4f},{int(gs_te[i])},{int(prob_te[i].argmax())}"
            f.write(base_row + "\n")
    # mae_pos is None when the test pool has no tc>0 rows (e.g. a negatives-only
    # holdout) — don't crash the summary print after all seeds trained.
    _sc_str = f"{mae_pos:.2f}" if mae_pos is not None else "n/a"
    print(f"[finetune] {len(seed_preds)}-seed ensemble: MAE {metrics['head']['overall']['mae_K']:.2f} K "
          f"(SC-only {_sc_str} over {int(_pos.sum())}; {int((~_pos).sum())} tc=0 rows) | "
          f"RMSE {rmse:.2f} | MSLE {msle:.3f} | cuprate MAE {metrics['head'].get('family/Cuprate',{}).get('mae_K',float('nan')):.2f} "
          f"[split={metrics['split_group']}, loss={loss_type}, unfreeze={mode}]")
    print(f"run dir: {out_dir}")
    return metrics
