"""Shared, model-agnostic training infrastructure.

Moved out of models/MPNN/MPNNMain.py so any model package (MPNN, GPSTransformer)
can reuse it without depending on another model subfolder. Holds the regression
training loop + the leaf utilities (target normalizer, meters, device move,
LR warmup / gradient clipping, checkpointing). The classification loop stays in
MPNNMain for now (only the MPNN baseline uses it).

`run_regression` takes the model as an argument — it never constructs one — so it
is fully model-agnostic; each model package supplies its own trainer entry that
builds its model and calls in here.
"""

import csv
import os
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.autograd import Variable


def _autocast(args):
    """bf16 autocast context when `amp` is set and running on CUDA, else a no-op.

    bf16 (not fp16) so no GradScaler is needed — its exponent range matches fp32,
    so gradients can't underflow. Opt-in via the config `amp` flag; off-by-default
    keeps existing runs (and the MPNN trainer) bit-for-bit unchanged. Halves the
    big activation tensors (e.g. the GPS poly attention's (N, heads, M, M))."""
    if args.get("amp") and args.get("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


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


def _to_input_var(input_batch, cuda):
    """Move a collated input tuple onto the device, arity-agnostically: every tensor
    element is moved (non_blocking), non-tensors (e.g. the python int n_crystals)
    pass through. Handles both the 8-element MPNN batch and the 10-element GPS batch
    (which appends frac_coords + lattice for the distance bias)."""
    if not cuda:
        return tuple(input_batch)
    return tuple(x.cuda(non_blocking=True) if torch.is_tensor(x) else x
                 for x in input_batch)


def _resolve_warmup_steps(args, steps_per_epoch):
    """Warmup length in optimizer steps from the config.

    `warmup_steps` wins if given; else `warmup_epochs * steps_per_epoch`; else 0
    (no warmup — existing configs are unchanged, bitwise-identically). Warmup is
    the highest-value stabilizer for the deeper/wider set-transformer: a cold
    start at full AdamW LR (no warmup, no clip) was what trapped the 642k-param
    run in a bad basin (train MAE stuck ~4x above the 126k model).
    """
    if args.get("warmup_steps"):
        return int(args["warmup_steps"])
    if args.get("warmup_epochs"):
        return int(round(float(args["warmup_epochs"]) * steps_per_epoch))
    return 0


def _warmup_and_step(optimizer, model, args, base_lr, warmup_steps, global_step):
    """Clip grads (if `grad_clip` > 0), apply linear LR warmup (if active), step.

    Call AFTER loss.backward(), in place of a bare optimizer.step(). Warmup
    linearly ramps every param group from 0 -> base_lr over the first
    `warmup_steps` steps; once past warmup it leaves the LR for MultiStepLR/SWALR
    to own (warmup finishes within epoch 0 at the usual sub-epoch lengths, and
    MultiStepLR holds base_lr until its first milestone, so the two never fight).
    """
    grad_clip = float(args.get("grad_clip", 0) or 0)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if warmup_steps and global_step < warmup_steps:
        scale = (global_step + 1) / warmup_steps
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * scale
    optimizer.step()


def _train(loader, model, criterion, optimizer, epoch, normalizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    model.train()
    end = time.time()

    steps_per_epoch = len(loader)
    warmup_steps = _resolve_warmup_steps(args, steps_per_epoch)
    base_lr = args["learning_rate"]

    for i, (input_batch, target, _lab, _) in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = _to_input_var(input_batch, args["cuda"])

        target_normed = normalizer.norm(target)
        target_var = Variable(target_normed.cuda(non_blocking=True) if args["cuda"] else target_normed)

        with _autocast(args):
            output = model(*input_var)
            loss = criterion(output, target_var)
        # backward()/clip/step stay OUTSIDE autocast and in fp32 (bf16 needs no
        # GradScaler); metric below casts output to fp32 so logs aren't bf16-noisy.

        # Metric accumulation stays ON-DEVICE and graph-free (see the original
        # MPNN trainer notes): denorm(target_var) recovers the raw target
        # on-device, so values materialize only at print time / epoch end.
        with torch.no_grad():
            mae_error = (normalizer.denorm(output.detach().float())
                         - normalizer.denorm(target_var.detach())).abs().mean()
        losses.update(loss.detach(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        _warmup_and_step(optimizer, model, args, base_lr, warmup_steps,
                         epoch * steps_per_epoch + i)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.get("print_split", 10) == 0:
            print(f"Epoch: [{epoch}][{i}/{len(loader)}]\t"
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                  f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                  f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})")

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

            with _autocast(args):
                output = model(*input_var)
                loss = criterion(output, target_var)
            output = output.float()    # back to fp32 for the metric + saved predictions

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

    return float(mae_errors.avg), float(losses.avg)


def run_regression(args, model, criterion, optimizer, scheduler, loaders, normalizer):
    """Model-agnostic regression training loop (SWA-aware, telemetry, test eval).

    The model is supplied by the caller; nothing here is MPNN- or GPS-specific.
    """
    best_mae_error = float("inf")
    train_losses, val_losses = [], []

    # Stochastic Weight Averaging: after `swa_start`, average the weights the
    # optimizer visits under a low constant LR (SWALR). LayerNorm-only model -> no
    # update_bn pass needed before using the averaged weights.
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
    val_real = loaders["val_realistic"]
    val_bal = loaders["val_balanced"]
    has_nonsc = loaders["split_sizes"]["val_nonsc"] > 0

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
        model.load_state_dict(swa_model.module.state_dict())
        torch.save({"state_dict": model.state_dict(),
                    "normalizer": normalizer.state_dict(), "args": args},
                   args["out_file"] + "_swa.pth.tar")
        print("Using SWA-averaged weights for the test evaluation "
              f"(saved {os.path.basename(args['out_file'])}_swa.pth.tar)")
    else:
        best_ckpt = torch.load(args["out_file"] + "_model_best.pth.tar")
        model.load_state_dict(best_ckpt["state_dict"])
    real_mae, _ = _validate(loaders["test_realistic"], model, criterion,
                            normalizer, args, test=True)
    if loaders["split_sizes"]["test_nonsc"] > 0:
        bal_test_mae, _ = _validate(loaders["test_balanced"], model, criterion,
                                    normalizer, args, test=True, tag="balanced")
        print(f" ** Test MAE (realistic): {real_mae:.3f}  (balanced): {bal_test_mae:.3f}")
    else:
        print(f" ** Test MAE: {real_mae:.3f}")
