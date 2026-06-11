import csv
import datetime
import json
import os
import shutil
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from MPNNData import (load_cif_dataset, collate_pool, get_sc_nonsc_loaders,
                      compute_feature_stats, resolve_split_by)
from MPNNModel import CrystalMPNN


def _parse_ratio(v):
    """Parse the SC_to_non_SC_ratio config value into a float.

    Accepts a number, or a string ("inf"/"infinity"/"none"/""), or None/omitted.
    Anything meaning "no non-SC" maps to float('inf') (the default).
    """
    if v is None:
        return float("inf")
    if isinstance(v, str):
        if v.strip().lower() in ("inf", "infinity", "none", ""):
            return float("inf")
        return float(v)
    return float(v)

best_mae_error = 1e10


def _write_run_metadata(run_dir, args, model, loaders, dataset, classification,
                        feature_dims):
    """Write run_dir/metadata.json: resolved hyperparameters + model-size stats.

    'params per sample' uses the effective per-epoch training set (all SC-train
    plus the non-SC actually sampled each epoch) — the count that bears on
    overfitting per epoch.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ss = loaders["split_sizes"]
    eff_train = ss["train_sc"] + ss["train_nonsc_per_epoch"]
    node_dim, edge_dim, poly_dim = feature_dims

    meta = {
        "run_id": args.get("run_id"),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_type": args.get("run_tag", "MPNN"),
        "task": "classification" if classification else "regression",
        "config_file": os.path.abspath(sys.argv[1]),
        "feature_dims": {"node": int(node_dim), "edge": int(edge_dim), "poly": int(poly_dim)},
        "architecture": {
            "atom_feat_len": args.get("atom_feat_len", 64),
            "edge_hidden_dim": args.get("edge_hidden_dim", 128),
            "n_conv": args.get("n_conv", 3),
            "h_feat_len": args.get("h_feat_len", 128),
            "n_hidden": args.get("n_hidden", 1),
            "edge_aggregation": args.get("edge_aggregation", args.get("aggregation", "ecn_weighted")),
            "atom_pooling": args.get("atom_pooling", "mean"),
            "set2set_steps": args.get("set2set_steps", 3),
            "use_poly_edges": args.get("use_poly_edges", True),
            "poly_fusion": args.get("poly_fusion", "sum"),
            "use_coord_magnitude": args.get("use_coord_magnitude", False),
            "set_transformer_heads": args.get("set_transformer_heads", 4),
            "normalize_features": args.get("normalize_features", True),
        },
        "training": {
            "optim": args.get("optim", "SGD"),
            "learning_rate": args.get("learning_rate"),
            "weight_decay": args.get("weight_decay", 0),
            "lr_milestones": args.get("lr_milestones", [100]),
            "epochs": args.get("epochs"),
            "batch_size": args.get("batch_size"),
            "target_transform": args.get("target_transform", "none"),
            "sc_to_non_sc_ratio": str(_parse_ratio(args.get("SC_to_non_SC_ratio"))),
            "sc_class_weight": args.get("sc_class_weight", 1.0),
            "sc_decision_threshold": args.get("sc_decision_threshold", 0.5),
            "sc_fbeta": args.get("sc_fbeta", 1.0),
            "selection_metric": args.get("selection_metric", "auc"),
        },
        "dataset": {
            "index_path": args.get("index_path"),
            "target_column": getattr(dataset, "target_column", args.get("target_column")),
            "total_indexed": len(dataset),
            "split_by": args.get("split_by", "frame"),
            "split_sizes": ss,
        },
        "model_size": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "effective_train_samples_per_epoch": eff_train,
            "params_per_train_sample": round(total_params / max(1, eff_train), 2),
        },
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model: {total_params:,} params "
          f"({meta['model_size']['params_per_train_sample']} per train sample, "
          f"{eff_train} effective train samples/epoch)")


def main():
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        warnings.warn("Usage: MPNNMain.py <config.json>")
        return -1

    with open(sys.argv[1]) as f:
        args = json.load(f)

    classification = args.get("task", "regression") == "classification"

    # --- Per-run output directory: model_data/<date>/<run_tag>/<run_id>/ ---
    # Each run is self-contained: a copy of the config, a metadata.json, and all
    # model artifacts (<out_file>_*) land here. run_tag defaults to "MPNN"
    # (use "Orig" for the baseline CGCNN). The dir is nested under <date>/<run_tag>/
    # so model_data/ stays navigable as runs pile up; the leaf keeps the full
    # run_id (tag+timestamp) so it's still self-describing in isolation and
    # scripts/reorg_runs.py can migrate older flat runs to the same layout.
    run_tag = args.get("run_tag", "MPNN")
    model_data_dir = args.get("model_data_dir", "model_data")
    now = datetime.datetime.now()
    run_id = f"{run_tag}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = os.path.join(model_data_dir, now.strftime('%Y-%m-%d'), run_tag, run_id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(sys.argv[1], os.path.join(run_dir, "config.json"))
    # Redirect every artifact (<out_file>_epoch_log.csv, _model_best.pth.tar, …)
    # into the run dir, keeping the configured basename as the file prefix.
    out_base = os.path.basename(args.get("out_file", "result")) or "result"
    args["out_file"] = os.path.join(run_dir, out_base)
    args["run_id"] = run_id
    print(f"Run output dir: {run_dir}")

    # The set-transformer aggregation needs the per-atom bond-angle matrix; build it
    # only for that variant (it costs extra per-sample bytes/CPU otherwise).
    _edge_agg = args.get("edge_aggregation", args.get("aggregation", "ecn_weighted"))
    # Factory: a pack directory (MPNNPack) and an index pickle (CIFDataV4) are
    # interchangeable here — identical interface and sample tensors.
    dataset = load_cif_dataset(
        args["index_path"],
        max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        graph_cache_size=args.get("graph_cache_size", 4096),
        target_column=args.get("target_column"),
        use_bond_angles=args.get("use_bond_angles", False),
        use_poly_edges=args.get("use_poly_edges", True),
        build_angle_bias=(_edge_agg == "set_transformer"),
    )

    # Optionally prebuild every sample once and keep it resident — see
    # CIFDataV4.prebuild. This removes the per-epoch JSON-read + sort/pad/stack so
    # the GPU stops starving on the input pipeline. `prebuild_device` picks where
    # the built dataset lives:
    #   "cpu"  (default): one copy in RAM; batches are still copied to the GPU each
    #          step, but that copy is a few MB, pinned + non_blocking, overlapped
    #          with compute (sub-ms). Fastest startup, no VRAM cost, fits any model.
    #   "cuda": resident in VRAM — removes even that per-batch copy, but startup is
    #          slower (~245k tiny H2D copies for ~49k crystals) and it spends VRAM.
    #   "auto": "cuda" if it fits with headroom for the model, else "cpu".
    # Training throughput is essentially identical between cpu/cuda (both serve full
    # cache hits); "cpu" is the recommended default. Falls back to "cpu" with no GPU.
    prebuilt_on_gpu = False
    if args.get("prebuild_dataset", False) and getattr(dataset, "is_packed", False):
        print(">> prebuild_dataset requested but dataset is packed — skipping "
              "(memmap reads are already fast; workers stay enabled).")
    elif args.get("prebuild_dataset", False):
        want = str(args.get("prebuild_device", "cpu")).lower()
        device = "cpu"
        if want in ("cuda", "auto") and torch.cuda.is_available():
            # Estimate the resident size from a spread of real samples (crystal
            # sizes vary) and check it fits with headroom for model + activations +
            # optimizer state; otherwise fall back to CPU RAM.
            probe = list(range(0, len(dataset), max(1, len(dataset) // 64)))
            def _bytes(s):
                feats, tgt, lbl, _ = s
                return (sum(x.element_size() * x.nelement() for x in feats)
                        + tgt.element_size() * tgt.nelement()
                        + lbl.element_size() * lbl.nelement())
            need = sum(_bytes(dataset[i]) for i in probe) / len(probe) * len(dataset)
            free, _total = torch.cuda.mem_get_info()
            if need < 0.75 * free:
                device = "cuda"
            else:
                print(f">> Prebuilt dataset ~{need/1e9:.1f} GB exceeds 75% of "
                      f"{free/1e9:.1f} GB free VRAM; prebuilding to CPU RAM instead.")
        elif want == "cuda" and not torch.cuda.is_available():
            print(">> prebuild_device='cuda' but no GPU available; using CPU RAM.")
        print(f">> Prebuilding {len(dataset)} samples on {device} "
              f"(single-process; ~one-time)...")
        t0 = time.time()
        dataset.prebuild(device=device)
        prebuilt_on_gpu = (device == "cuda")
        print(f">> Prebuild done in {time.time() - t0:.1f}s "
              f"(resident on {device}); dataloader workers -> 0")

    # With a prebuilt dataset there's no build work to parallelize, and GPU-resident
    # tensors can't cross a DataLoader worker boundary, so force single-process,
    # in-loop loading. pin_memory only helps a CPU->GPU copy, which doesn't happen
    # once the data already lives on the GPU.
    eff_workers = (0 if getattr(dataset, "_prebuilt", None) is not None
                   else args.get("num_workers", 0))
    eff_pin = torch.cuda.is_available() and not prebuilt_on_gpu

    # SC:non-SC ratio controls how many non-SC samples are drawn per epoch
    # (default inf -> none). Both tasks share the same stratified split + per-
    # epoch resampled-non-SC train loader + realistic/balanced val/test sets.
    sc_to_nonsc_ratio = _parse_ratio(args.get("SC_to_non_SC_ratio"))
    # Split granularity (see MPNNData.resolve_split_by — shared with eval_test).
    split_by = resolve_split_by(args.get("split_by"), dataset)
    args["split_by"] = split_by   # resolved value -> recorded in metadata.json
    loaders = get_sc_nonsc_loaders(
        dataset,
        batch_size=args["batch_size"],
        val_ratio=args["val_ratio"],
        test_ratio=args["test_ratio"],
        sc_to_nonsc_ratio=sc_to_nonsc_ratio,
        num_workers=eff_workers,
        pin_memory=eff_pin,
        seed=args.get("split_seed", 123),
        split_by=split_by,
    )
    print(f"SC_to_non_SC_ratio={sc_to_nonsc_ratio}, split_by={split_by} "
          f"-> split sizes:", loaders["split_sizes"])

    normalizer = None
    if not classification:
        # Normalizer for target (T_c), computed over the per-epoch training
        # composition: all SC-train targets plus the n_nonsc non-SC targets
        # that get sampled in. Read straight from the index
        # (dataset.data[i] = (id, value, graph_path, label)) — no graph loading.
        sc_idx = loaders["train_sc_idx"]
        ns_idx = loaders["train_nonsc_idx"]
        n_nonsc = loaders["n_nonsc_per_epoch"]
        train_targets = (
            [float(dataset.data[i][1]) for i in sc_idx]
            + [float(dataset.data[i][1]) for i in ns_idx[:n_nonsc]]
        )
        sample_targets = torch.tensor(train_targets, dtype=torch.float32)
        normalizer = Normalizer(sample_targets, transform=args.get("target_transform", "none"))

    # Infer feature dims from first sample
    (sample_atom, sample_nbr, _, sample_poly, _, _), _, _lab, _ = dataset[0]
    orig_atom_fea_len = sample_atom.shape[-1]
    nbr_fea_len = sample_nbr.shape[-1]
    poly_fea_len = sample_poly.shape[-1]

    # Seed torch before constructing the model so weight init is reproducible and,
    # for ensembling, distinct per member (vary `model_seed` across runs to get
    # diverse members; otherwise MPNNData's module-level manual_seed(0) makes every
    # run init identically). Only set when provided, to preserve old behaviour.
    if args.get("model_seed") is not None:
        torch.manual_seed(int(args["model_seed"]))

    model = CrystalMPNN(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        poly_fea_len=poly_fea_len,
        atom_fea_len=args.get("atom_feat_len", 64),
        edge_hidden_dim=args.get("edge_hidden_dim", 128),
        n_conv=args.get("n_conv", 3),
        h_fea_len=args.get("h_feat_len", 128),
        n_h=args.get("n_hidden", 1),
        # `edge_aggregation` is the descriptive name; fall back to the legacy
        # `aggregation` key so older configs keep working.
        edge_aggregation=args.get("edge_aggregation", args.get("aggregation", "ecn_weighted")),
        classification=classification,
        use_poly_edges=args.get("use_poly_edges", True),
        atom_pooling=args.get("atom_pooling", "mean"),
        set2set_steps=args.get("set2set_steps", 3),
        # Off (0.0) by default for regression; classification keeps its historical
        # 0.5 unless the config overrides it.
        dropout=args.get("dropout", 0.5 if classification else 0.0),
        # Opt-in: re-inject per-atom total coordination strength into each message
        # (the intensive weighted aggregation otherwise discards it). Off by default
        # so existing configs/param counts are unchanged.
        use_coord_magnitude=args.get("use_coord_magnitude", False),
        # Heads for the 'set_transformer' edge aggregation (idea A); ignored by the
        # ecn_weighted / attention variants.
        set_transformer_heads=args.get("set_transformer_heads", 4),
        # How bond and poly messages combine ('sum' | 'gate'); only meaningful
        # with use_poly_edges.
        poly_fusion=args.get("poly_fusion", "sum"),
    )

    # Per-feature input standardization, computed on the training pool only
    # (SC-train + the non-SC pool the sampler draws from). Installed as model
    # buffers before .cuda() so they move to the device with the model and are
    # saved/restored with the checkpoint.
    if args.get("normalize_features", True):
        sc_idx = loaders["train_sc_idx"]
        ns_idx = loaders["train_nonsc_idx"]
        stat_idx = list(sc_idx) + (list(ns_idx) if loaders["n_nonsc_per_epoch"] > 0 else [])
        stats = compute_feature_stats(
            dataset, stat_idx,
            max_graphs=args.get("feature_stat_graphs", 4000),
            seed=args.get("split_seed", 123),
        )
        model.set_feature_stats(
            stats["node"], stats["edge"],
            stats["poly"] if model.use_poly_edges else None,
        )
        print("Installed per-feature input normalization "
              f"(from {min(len(stat_idx), args.get('feature_stat_graphs', 4000))} train graphs)")

    # Record run metadata (hyperparameters + model size) before training so it
    # exists even if the run is interrupted.
    _write_run_metadata(run_dir, args, model, loaders, dataset, classification,
                        feature_dims=(orig_atom_fea_len, nbr_fea_len, poly_fea_len))

    args["cuda"] = torch.cuda.is_available()
    if args["cuda"]:
        model.cuda()

    if classification:
        # Up-weight the SC (positive) class so missing a real superconductor
        # (false negative) is penalized more than a false alarm. index 0=non-SC,
        # 1=SC. sc_class_weight > 1 trades precision for recall.
        sc_w = float(args.get("sc_class_weight", 1.0))
        weight = torch.tensor([1.0, sc_w])
        if args["cuda"]:
            weight = weight.cuda()
        criterion = nn.NLLLoss(weight=weight)
    else:
        criterion = nn.L1Loss()

    optim_name = args.get("optim", "SGD")
    if optim_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args["learning_rate"],
                              momentum=args.get("momentum", 0.9),
                              weight_decay=args.get("weight_decay", 0))
    elif optim_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"],
                               weight_decay=args.get("weight_decay", 0))
    elif optim_name == "AdamW":
        # Decoupled weight decay — more effective regularization than Adam's
        # coupled L2 when weight_decay > 0 (identical to Adam when it's 0).
        optimizer = optim.AdamW(model.parameters(), lr=args["learning_rate"],
                                weight_decay=args.get("weight_decay", 0))
    else:
        raise ValueError(f"Unsupported optim: {optim_name}")

    scheduler = MultiStepLR(optimizer, milestones=args.get("lr_milestones", [100]), gamma=0.1)

    if classification:
        _run_classification(args, model, criterion, optimizer, scheduler, loaders)
    else:
        _run_regression(args, model, criterion, optimizer, scheduler, loaders, normalizer)


def _run_regression(args, model, criterion, optimizer, scheduler, loaders, normalizer):
    global best_mae_error
    train_losses, val_losses = [], []

    # Stochastic Weight Averaging: after `swa_start`, average the weights the
    # optimizer visits under a low constant LR (SWALR). The averaged weights tend
    # to sit in a flatter minimum that generalizes better — a cheap win in exactly
    # the post-plateau regime this model shows. The model uses LayerNorm (no
    # BatchNorm), so no update_bn pass is needed before using the averaged weights.
    from torch.optim.swa_utils import AveragedModel, SWALR
    swa_on = bool(args.get("swa", False))
    swa_model = swa_scheduler = None
    if swa_on:
        swa_model = AveragedModel(model)
        swa_start = int(args.get("swa_start", int(0.75 * args["epochs"])))
        swa_scheduler = SWALR(optimizer, swa_lr=float(args.get("swa_lr", args["learning_rate"] * 0.05)))
        print(f"SWA enabled: averaging from epoch {swa_start} at swa_lr="
              f"{args.get('swa_lr', args['learning_rate'] * 0.05)}")

    train_loader = loaders["train"]
    # Model selection on the realistic val MAE; balanced val MAE logged too.
    # When no non-SC are present the balanced set is identical to the realistic
    # one, so skip the redundant pass and mirror the realistic MAE.
    val_real = loaders["val_realistic"]
    val_bal = loaders["val_balanced"]
    has_nonsc = loaders["split_sizes"]["val_nonsc"] > 0

    # per-epoch telemetry log (mirrors CGCNNMain.py). Resource columns: GPU
    # utilization/memory + process-tree CPU/RSS sampled by a background thread —
    # so over/under-allocation (idle GPU, starved dataloader workers) is visible
    # per epoch instead of requiring a separate profiling run.
    from resmon import ResourceMonitor
    monitor = ResourceMonitor(interval=2.0).start()
    print(ResourceMonitor.describe())
    epoch_log_file = open(args["out_file"] + "_epoch_log.csv", "w", newline="")
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(["epoch", "train_loss", "train_mae", "val_loss",
                           "val_mae", "val_bal_mae", "lr", "epoch_time_sec",
                           "train_time_sec", "data_time_sec",
                           "gpu_util_pct", "gpu_mem_gb", "cpu_pct", "rss_gb",
                           "is_best"])

    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_mae, data_time_s, train_time_s = _train(
            train_loader, model, criterion, optimizer, epoch, normalizer, args)
        mae_error, val_loss = _validate(val_real, model, criterion, normalizer, args)
        bal_mae = _validate(val_bal, model, criterion, normalizer, args)[0] if has_nonsc else mae_error

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if mae_error != mae_error:
            print("Exit due to NaN")
            sys.exit(1)

        # During the SWA phase, accumulate the averaged weights and hold a low
        # constant LR; otherwise follow the normal MultiStepLR schedule.
        if swa_on and epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        _save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_mae_error": best_mae_error,
            "optimizer": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "args": args,
        }, is_best, args["out_file"])

        epoch_time = time.time() - epoch_start
        res = monitor.epoch_stats()
        epoch_logger.writerow([epoch, float(train_loss), float(train_mae),
                               float(val_loss), float(mae_error), float(bal_mae), lr,
                               epoch_time, round(train_time_s, 2), round(data_time_s, 2),
                               res["gpu_util_pct"], res["gpu_mem_gb"],
                               res["cpu_pct"], res["rss_gb"],
                               int(is_best)])
        epoch_log_file.flush()
        if epoch % args.get("print_split", 10) == 0:
            print(f">> epoch {epoch}: {epoch_time:.1f}s total "
                  f"(train {train_time_s:.1f}s, of which data-wait {data_time_s:.1f}s) | "
                  f"GPU {_fmt_res(res['gpu_util_pct'])}% {_fmt_res(res['gpu_mem_gb'])}GB | "
                  f"CPU {_fmt_res(res['cpu_pct'])}% RSS {_fmt_res(res['rss_gb'])}GB")

    monitor.stop()
    epoch_log_file.close()

    np.save(args["out_file"] + "_losstrain.csv", np.array(train_losses))
    np.save(args["out_file"] + "_lossval.csv", np.array(val_losses))

    print("-" * 50 + "\nEvaluating on test set")
    if swa_on:
        # Use the SWA-averaged weights (LayerNorm -> no update_bn needed). Saved
        # separately so the val-best checkpoint is still available for comparison.
        model.load_state_dict(swa_model.module.state_dict())
        torch.save({"state_dict": model.state_dict(),
                    "normalizer": normalizer.state_dict(), "args": args},
                   args["out_file"] + "_swa.pth.tar")
        print("Using SWA-averaged weights for the test evaluation "
              f"(saved {os.path.basename(args['out_file'])}_swa.pth.tar)")
    else:
        best_ckpt = torch.load(args["out_file"] + "_model_best.pth.tar")
        model.load_state_dict(best_ckpt["state_dict"])
    # Evaluate the realistic test set (writes the canonical out_file.csv
    # predictions). Only evaluate the balanced set when non-SC are present —
    # otherwise it is identical to the realistic set.
    real_mae, _ = _validate(loaders["test_realistic"], model, criterion,
                            normalizer, args, test=True)
    if loaders["split_sizes"]["test_nonsc"] > 0:
        bal_test_mae, _ = _validate(loaders["test_balanced"], model, criterion,
                                    normalizer, args, test=True, tag="balanced")
        print(f" ** Test MAE (realistic): {real_mae:.3f}  (balanced): {bal_test_mae:.3f}")
    else:
        print(f" ** Test MAE: {real_mae:.3f}")


def _run_classification(args, model, criterion, optimizer, scheduler, loaders):
    """Train + evaluate the MPNN as a Stage-1 SC/non-SC classifier.

    Model selection uses validation AUC on the realistic (imbalanced) split;
    metrics logged on both realistic and balanced val; test evaluated on both.
    """
    best_score = -1.0
    # Metric used to pick the best checkpoint. Default AUC (threshold-free). For
    # a recall-priority model set selection_metric="fbeta" with sc_fbeta=2.
    selection_metric = args.get("selection_metric", "auc")
    train_loader = loaders["train"]
    val_real = loaders["val_realistic"]
    val_bal = loaders["val_balanced"]

    from resmon import ResourceMonitor
    monitor = ResourceMonitor(interval=2.0).start()
    print(ResourceMonitor.describe())
    epoch_log_file = open(args["out_file"] + "_epoch_log.csv", "w", newline="")
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(["epoch", "train_loss", "train_acc", "val_loss",
                           "val_acc", "val_precision", "val_recall", "val_f1",
                           "val_fbeta", "val_auc", "val_bal_acc", "lr",
                           "epoch_time_sec", "train_time_sec", "data_time_sec",
                           "gpu_util_pct", "gpu_mem_gb", "cpu_pct", "rss_gb",
                           "is_best"])

    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc, data_time_s, train_time_s = _train_classification(
            train_loader, model, criterion, optimizer, epoch, args)
        val_metrics, val_loss = _validate_classification(val_real, model, criterion, args)
        bal_metrics, _ = _validate_classification(val_bal, model, criterion, args)

        if val_loss != val_loss:
            print("Exit due to NaN")
            sys.exit(1)

        scheduler.step()

        val_auc = val_metrics["auc"]
        # Selection score = configured metric; fall back to f1 if it's nan
        # (e.g. AUC on a single-class val batch).
        select_metric = val_metrics.get(selection_metric, float("nan"))
        if select_metric != select_metric:
            select_metric = val_metrics["f1"]
        is_best = select_metric > best_score
        best_score = max(select_metric, best_score)
        _save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_score": best_score,
            "selection_metric": selection_metric,
            "optimizer": optimizer.state_dict(),
            "args": args,
        }, is_best, args["out_file"])

        epoch_time = time.time() - epoch_start
        res = monitor.epoch_stats()
        epoch_logger.writerow([epoch, float(train_loss), float(train_acc),
                               float(val_loss), float(val_metrics["accuracy"]),
                               float(val_metrics["precision"]), float(val_metrics["recall"]),
                               float(val_metrics["f1"]), float(val_metrics["fbeta"]),
                               float(val_auc), float(bal_metrics["accuracy"]), lr,
                               epoch_time, round(train_time_s, 2), round(data_time_s, 2),
                               res["gpu_util_pct"], res["gpu_mem_gb"],
                               res["cpu_pct"], res["rss_gb"],
                               int(is_best)])
        epoch_log_file.flush()
        if epoch % args.get("print_split", 10) == 0:
            print(f">> epoch {epoch}: {epoch_time:.1f}s total "
                  f"(train {train_time_s:.1f}s, of which data-wait {data_time_s:.1f}s) | "
                  f"GPU {_fmt_res(res['gpu_util_pct'])}% {_fmt_res(res['gpu_mem_gb'])}GB | "
                  f"CPU {_fmt_res(res['cpu_pct'])}% RSS {_fmt_res(res['rss_gb'])}GB")

    monitor.stop()
    epoch_log_file.close()

    print("-" * 50 + "\nEvaluating classifier on test set")
    best_ckpt = torch.load(args["out_file"] + "_model_best.pth.tar")
    model.load_state_dict(best_ckpt["state_dict"])
    test_real, _ = _validate_classification(loaders["test_realistic"], model, criterion, args, test=True, tag="realistic")
    test_bal, _ = _validate_classification(loaders["test_balanced"], model, criterion, args, test=True, tag="balanced")
    print(" ** Test (realistic):", {k: round(v, 4) for k, v in test_real.items()})
    print(" ** Test (balanced): ", {k: round(v, 4) for k, v in test_bal.items()})


def _train(loader, model, criterion, optimizer, epoch, normalizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    model.train()
    end = time.time()

    for i, (input_batch, target, _lab, _) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = _to_input_var(input_batch, args["cuda"])

        target_normed = normalizer.norm(target)
        target_var = Variable(target_normed.cuda(non_blocking=True) if args["cuda"] else target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        # Metric accumulation stays ON-DEVICE and graph-free. The previous
        # .cpu() pulls here forced a full GPU sync twice per batch (~22k pipeline
        # stalls per MPtrj epoch). denorm(target_var) recovers the raw target
        # on-device (denorm∘norm == identity up to fp32 rounding), avoiding a
        # second host->device copy. Values materialize only when formatted at
        # print time (every print_split batches) and at the epoch-end float() —
        # so per-batch "Time" now reads as async launch time, with sync cost
        # landing on the print batches; epoch totals are unaffected.
        with torch.no_grad():
            mae_error = (normalizer.denorm(output.detach())
                         - normalizer.denorm(target_var.detach())).abs().mean()
        losses.update(loss.detach(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.get("print_split", 10) == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})")

    # data_time.sum = seconds the loop spent WAITING on the DataLoader (collate +
    # any worker fetch); batch_time.sum − data_time.sum ≈ model compute + syncs.
    # float() here is the single end-of-epoch GPU sync for the accumulated metrics.
    return float(losses.avg), float(mae_errors.avg), data_time.sum, batch_time.sum


def _validate(loader, model, criterion, normalizer, args, test=False, tag=""):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    test_targets, test_preds, test_cif_ids = [], [], []
    model.eval()
    end = time.time()

    for i, (input_batch, target, _lab, batch_cif_ids) in enumerate(loader):
        with torch.no_grad():
            input_var = _to_input_var(input_batch, args["cuda"])

            target_normed = normalizer.norm(target)
            target_var = Variable(target_normed.cuda(non_blocking=True) if args["cuda"] else target_normed)

            output = model(*input_var)
            loss = criterion(output, target_var)

        # On-device metric accumulation (no per-batch sync) — see _train. The
        # test branch still pulls predictions to the CPU (needed for the results
        # CSV), but test runs once, not every epoch.
        with torch.no_grad():
            mae_error = (normalizer.denorm(output.detach())
                         - normalizer.denorm(target_var.detach())).abs().mean()
        losses.update(loss.detach(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        if test:
            test_preds += normalizer.denorm(output.data.cpu()).view(-1).tolist()
            test_targets += target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.get("print_split", 10) == 0:
            print(f"  [{i}/{len(loader)}]  Loss {losses.val:.4f} ({losses.avg:.4f})  "
                  f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})")

    label = "**" if test else "*"
    print(f" {label} MAE {mae_errors.avg:.3f}")

    if test:
        results_path = args["out_file"] + ("_test_%s.csv" % tag if tag else ".csv")
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            for cif_id, tgt, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, tgt, pred))

    # Single end-of-pass sync; downstream comparisons/checkpoints get plain floats.
    return float(mae_errors.avg), float(losses.avg)


def _to_input_var(input_batch, cuda):
    """Move a collated input tuple onto the right device.

    Layout: (atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx,
             nbr_angle, crystal_atom_idx).
    """
    (atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx,
     nbr_angle, crystal_atom_idx) = input_batch
    if cuda:
        return (
            Variable(atom_fea.cuda(non_blocking=True)),
            Variable(nbr_fea.cuda(non_blocking=True)),
            nbr_fea_idx.cuda(non_blocking=True),
            Variable(poly_fea.cuda(non_blocking=True)),
            poly_fea_idx.cuda(non_blocking=True),
            nbr_angle.cuda(non_blocking=True),
            [idx.cuda(non_blocking=True) for idx in crystal_atom_idx],
        )
    return (Variable(atom_fea), Variable(nbr_fea), nbr_fea_idx,
            Variable(poly_fea), poly_fea_idx, nbr_angle, crystal_atom_idx)


def classification_metrics(log_probs, targets, threshold=0.5, beta=1.0):
    """Metrics for the SC/non-SC head. log_probs: (N,2) log-softmax CPU tensor;
    targets: (N,) long tensor.

    threshold : predict SC when p_sc >= threshold (default 0.5 = argmax).
                Lowering it raises recall at the cost of precision — useful when
                missing a superconductor is worse than a false positive.
    beta      : F-beta weighting (beta>1 favors recall). AUC is threshold-
                independent; nan if the set is single-class.
    """
    p_sc = np.exp(log_probs.numpy())[:, 1]
    pred_label = (p_sc >= threshold).astype(int)
    target_label = targets.numpy().reshape(-1)
    accuracy = metrics.accuracy_score(target_label, pred_label)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target_label, pred_label, average="binary", zero_division=0)
    fbeta = metrics.fbeta_score(target_label, pred_label, beta=beta, zero_division=0)
    try:
        auc = metrics.roc_auc_score(target_label, p_sc)
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": fscore, "fbeta": fbeta, "auc": auc}


def _train_classification(loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.train()
    end = time.time()

    for i, (input_batch, target, label, _) in enumerate(loader):
        data_time.update(time.time() - end)
        input_var = _to_input_var(input_batch, args["cuda"])
        target_var = Variable(label.cuda(non_blocking=True)) if args["cuda"] else Variable(label)

        output = model(*input_var)
        loss = criterion(output, target_var)

        pred = output.data.cpu().numpy().argmax(axis=1)
        acc = float((pred == label.cpu().numpy()).mean())  # label may be GPU-resident
        losses.update(loss.data.cpu().item(), label.size(0))
        accuracies.update(acc, label.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.get("print_split", 10) == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Acc {accuracies.val:.3f} ({accuracies.avg:.3f})")

    return losses.avg, accuracies.avg, data_time.sum, batch_time.sum


def _validate_classification(loader, model, criterion, args, test=False, tag=""):
    """Evaluate the classifier over a whole loader (metrics computed once on the
    full set). When test=True, writes (cif_id, true_label, p_sc) predictions."""
    losses = AverageMeter()
    model.eval()
    all_log_probs, all_labels, all_cif_ids = [], [], []

    for i, (input_batch, target, label, batch_cif_ids) in enumerate(loader):
        with torch.no_grad():
            input_var = _to_input_var(input_batch, args["cuda"])
            target_var = Variable(label.cuda(non_blocking=True)) if args["cuda"] else Variable(label)
            output = model(*input_var)
            loss = criterion(output, target_var)
        losses.update(loss.data.cpu().item(), label.size(0))
        all_log_probs.append(output.data.cpu())
        all_labels.append(label.cpu())   # may be GPU-resident; metrics/IO are CPU
        all_cif_ids += batch_cif_ids

    log_probs = torch.cat(all_log_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    m = classification_metrics(log_probs, labels,
                               threshold=args.get("sc_decision_threshold", 0.5),
                               beta=args.get("sc_fbeta", 1.0))

    if test:
        pos_prob = np.exp(log_probs.numpy())[:, 1]
        results_path = args["out_file"] + ("_test_%s.csv" % tag if tag else ".csv")
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["cif_id", "true_label", "p_sc"])
            for cid, tl, p in zip(all_cif_ids, labels.numpy().tolist(), pos_prob.tolist()):
                writer.writerow((cid, int(tl), p))

    return m, losses.avg


# --- Utilities (mirrors CGCNNMain.py) ---

class Normalizer:
    """Target normalizer with an optional monotone transform applied BEFORE the
    z-score (and inverted after denorm).

    transform="log1p" trains the regressor in log(1+T_c) space, which spreads
    out the densely-packed low-T_c region so the loss stops being dominated by
    a handful of high-T_c materials. T_c=0 maps to 0 (log1p(0)=0), so non-SC
    negatives are handled cleanly. norm() returns the transformed+standardized
    target (the training objective); denorm() inverts all the way back to real
    T_c, so reported MAE stays in physical units and is comparable across
    transforms.
    """

    def __init__(self, tensor, transform="none"):
        self.transform = transform
        t = self._fwd(tensor)
        self.mean = torch.mean(t)
        self.std = torch.std(t)

    def _fwd(self, x):
        return torch.log1p(x) if self.transform == "log1p" else x

    def _inv(self, y):
        return torch.expm1(y) if self.transform == "log1p" else y

    def norm(self, tensor):
        return (self._fwd(tensor) - self.mean) / self.std

    def denorm(self, normed_tensor):
        return self._inv(normed_tensor * self.std + self.mean)

    def state_dict(self):
        return {"mean": self.mean, "std": self.std, "transform": self.transform}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
        self.transform = state_dict.get("transform", "none")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _fmt_res(v):
    """Telemetry print formatting: '' (source unavailable) -> '?'; a legitimate
    0.0 reading (e.g. a fully idle GPU) must still print as 0.0."""
    return "?" if v == "" else v


def _save_checkpoint(state, is_best, filename):
    torch.save(state, filename + "_checkpoint.pth.tar")
    if is_best:
        shutil.copyfile(filename + "_checkpoint.pth.tar", filename + "_model_best.pth.tar")


if __name__ == "__main__":
    main()
