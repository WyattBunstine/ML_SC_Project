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

from MPNNData import (CIFDataV4, collate_pool, get_sc_nonsc_loaders,
                      compute_feature_stats, NODE_FEA_LEN, NBR_FEA_LEN)
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

    # --- Per-run output directory: model_data/<run_tag>_<timestamp>/ ---
    # Each run is self-contained: a copy of the config, a metadata.json, and all
    # model artifacts (<out_file>_*) land here. run_tag defaults to "MPNN"
    # (use "Orig" for the baseline CGCNN).
    run_tag = args.get("run_tag", "MPNN")
    model_data_dir = args.get("model_data_dir", "model_data")
    run_id = f"{run_tag}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = os.path.join(model_data_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(sys.argv[1], os.path.join(run_dir, "config.json"))
    # Redirect every artifact (<out_file>_epoch_log.csv, _model_best.pth.tar, …)
    # into the run dir, keeping the configured basename as the file prefix.
    out_base = os.path.basename(args.get("out_file", "result")) or "result"
    args["out_file"] = os.path.join(run_dir, out_base)
    args["run_id"] = run_id
    print(f"Run output dir: {run_dir}")

    dataset = CIFDataV4(
        index_path=args["index_path"],
        max_num_nbr=args.get("max_num_nbr", 14),
        max_num_poly_nbr=args.get("max_num_poly_nbr", 16),
        graph_cache_size=args.get("graph_cache_size", 4096),
        target_column=args.get("target_column"),
        use_bond_angles=args.get("use_bond_angles", False),
    )

    # SC:non-SC ratio controls how many non-SC samples are drawn per epoch
    # (default inf -> none). Both tasks share the same stratified split + per-
    # epoch resampled-non-SC train loader + realistic/balanced val/test sets.
    sc_to_nonsc_ratio = _parse_ratio(args.get("SC_to_non_SC_ratio"))
    loaders = get_sc_nonsc_loaders(
        dataset,
        batch_size=args["batch_size"],
        val_ratio=args["val_ratio"],
        test_ratio=args["test_ratio"],
        sc_to_nonsc_ratio=sc_to_nonsc_ratio,
        num_workers=args.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        seed=args.get("split_seed", 123),
    )
    print(f"SC_to_non_SC_ratio={sc_to_nonsc_ratio} -> split sizes:", loaders["split_sizes"])

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
    (sample_atom, sample_nbr, _, sample_poly, _), _, _lab, _ = dataset[0]
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

    # per-epoch telemetry log (mirrors CGCNNMain.py)
    epoch_log_file = open(args["out_file"] + "_epoch_log.csv", "w", newline="")
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(["epoch", "train_loss", "train_mae", "val_loss",
                           "val_mae", "val_bal_mae", "lr", "epoch_time_sec", "is_best"])

    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_mae = _train(train_loader, model, criterion, optimizer, epoch, normalizer, args)
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
        epoch_logger.writerow([epoch, float(train_loss), float(train_mae),
                               float(val_loss), float(mae_error), float(bal_mae), lr,
                               epoch_time, int(is_best)])
        epoch_log_file.flush()

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

    epoch_log_file = open(args["out_file"] + "_epoch_log.csv", "w", newline="")
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(["epoch", "train_loss", "train_acc", "val_loss",
                           "val_acc", "val_precision", "val_recall", "val_f1",
                           "val_fbeta", "val_auc", "val_bal_acc", "lr",
                           "epoch_time_sec", "is_best"])

    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_acc = _train_classification(train_loader, model, criterion, optimizer, epoch, args)
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
        epoch_logger.writerow([epoch, float(train_loss), float(train_acc),
                               float(val_loss), float(val_metrics["accuracy"]),
                               float(val_metrics["precision"]), float(val_metrics["recall"]),
                               float(val_metrics["f1"]), float(val_metrics["fbeta"]),
                               float(val_auc), float(bal_metrics["accuracy"]), lr,
                               epoch_time, int(is_best)])
        epoch_log_file.flush()

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

        mae_error = _mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.get("print_split", 10) == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})")

    return losses.avg, mae_errors.avg


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

        mae_error = _mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
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

    return mae_errors.avg, losses.avg


def _to_input_var(input_batch, cuda):
    """Move a collated input tuple onto the right device.

    Layout: (atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx,
             crystal_atom_idx).
    """
    atom_fea, nbr_fea, nbr_fea_idx, poly_fea, poly_fea_idx, crystal_atom_idx = input_batch
    if cuda:
        return (
            Variable(atom_fea.cuda(non_blocking=True)),
            Variable(nbr_fea.cuda(non_blocking=True)),
            nbr_fea_idx.cuda(non_blocking=True),
            Variable(poly_fea.cuda(non_blocking=True)),
            poly_fea_idx.cuda(non_blocking=True),
            [idx.cuda(non_blocking=True) for idx in crystal_atom_idx],
        )
    return (Variable(atom_fea), Variable(nbr_fea), nbr_fea_idx,
            Variable(poly_fea), poly_fea_idx, crystal_atom_idx)


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
    losses = AverageMeter()
    accuracies = AverageMeter()
    model.train()
    end = time.time()

    for i, (input_batch, target, label, _) in enumerate(loader):
        input_var = _to_input_var(input_batch, args["cuda"])
        target_var = Variable(label.cuda(non_blocking=True)) if args["cuda"] else Variable(label)

        output = model(*input_var)
        loss = criterion(output, target_var)

        pred = output.data.cpu().numpy().argmax(axis=1)
        acc = float((pred == label.numpy()).mean())
        losses.update(loss.data.cpu().item(), label.size(0))
        accuracies.update(acc, label.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.get("print_split", 10) == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"Acc {accuracies.val:.3f} ({accuracies.avg:.3f})")

    return losses.avg, accuracies.avg


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
        all_labels.append(label)
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


def _mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))


def _save_checkpoint(state, is_best, filename):
    torch.save(state, filename + "_checkpoint.pth.tar")
    if is_best:
        shutil.copyfile(filename + "_checkpoint.pth.tar", filename + "_model_best.pth.tar")


if __name__ == "__main__":
    main()
