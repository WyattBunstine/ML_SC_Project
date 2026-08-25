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

from dist_utils import all_reduce_grads, all_reduce_min_int, barrier  # noqa: E402


def _autocast(args):
    """bf16 autocast context when `amp` is set and running on CUDA, else a no-op.

    bf16 (not fp16) so no GradScaler is needed — its exponent range matches fp32,
    so gradients can't underflow. Opt-in via the config `amp` flag; off-by-default
    keeps existing runs (and the MPNN trainer) bit-for-bit unchanged. Halves the
    big activation tensors (e.g. the GPS poly attention's (N, heads, M, M))."""
    if args.get("amp") and args.get("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _setup_matmul_precision(args):
    """Enable TF32 matmuls on Ampere+ GPUs — a ~1.3-2x speedup on the matmul-heavy
    encoder (attention + FFNs + per-atom heads) for FREE. TF32 keeps full fp32
    dynamic RANGE and only rounds the matmul *inputs* to a 10-bit mantissa, so it is
    far gentler than the bf16 autocast we deliberately avoid on the conservative-force
    path (`amp=false` for multitask). On by default on CUDA; set `"tf32": false` to
    force exact fp32 (e.g. an A/B against `scripts/verify_tf32_forces.py`). No effect
    on CPU or the fp64 finite-difference force gate — TF32 is CUDA-matmul-only."""
    if not args.get("cuda"):
        return
    if args.get("tf32", True):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 matmuls ENABLED (float32_matmul_precision='high'); "
              "set \"tf32\": false in the config for exact fp32.")
    else:
        torch.set_float32_matmul_precision("highest")
        print("TF32 matmuls DISABLED (exact fp32).")


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
    _setup_matmul_precision(args)
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


# ============================ multitask pretraining ============================
# Conservative-autograd multi-target pretraining (energy/forces/stress/magmom/bandgap
# /dos). Entirely separate from run_regression above so the single-target MPNN/GPS path
# is untouched. The model returns a dict; forces/stress come from autograd of the
# extensive energy, so each step does a DOUBLE backward (the force/stress loss backprops
# through the model's internal autograd.grad). Run with amp=false (bf16 ruins force
# gradients) and the geometry collate (collate_pool_multitask).

_DEFAULT_LOSS_WEIGHTS = {"energy": 1.0, "forces": 10.0, "stress": 1.0,
                         "magmom": 1.0, "bandgap": 1.0, "dos": 1.0,
                         "eph_lambda": 1.0, "eph_wlog": 1.0,
                         "eph_a2f": 1.0, "ph_dos": 1.0}


def compute_target_stats(dataset, indices, max_samples=2000, seed=123):
    """Per-target scale (std over PRESENT values) for normalizing the multitask loss,
    from a sample of the MT training set. Only the std is needed (the mean cancels in
    the prediction-minus-target error). Absent targets are skipped."""
    import random
    idx = list(indices)
    if max_samples and len(idx) > max_samples:
        idx = random.Random(seed).sample(idx, max_samples)
    acc = {}
    for i in idx:
        _inp, tg, mk, _cid = dataset[i]
        for k, v in tg.items():
            if bool(mk[k]):
                acc.setdefault(k, []).append(v.reshape(-1))
    stats = {}
    for k, v in acc.items():
        s = float(torch.cat(v).std())          # std of a single present value is NaN
        stats[k] = max(s, 1e-6) if s == s else 1.0
    return stats


def _build_cart_strain(input_var):
    """Leaves the autograd forces/stress differentiate: the Cartesian position
    cart = frac @ lattice[seg] (forces = -dE/dcart) and a zero strain (stress =
    dE/dstrain). Built per batch so grads don't accumulate across steps."""
    seg, n_crystals, frac, lattice = input_var[6], input_var[7], input_var[8], input_var[9]
    cart = torch.einsum("ni,nij->nj", frac, lattice[seg]).detach().requires_grad_(True)
    strain = torch.zeros(n_crystals, 3, 3, device=cart.device, dtype=cart.dtype,
                         requires_grad=True)
    return cart, strain


def _move_target_dicts(targets, masks, cuda):
    if not cuda:
        return targets, masks
    return ({k: v.cuda(non_blocking=True) for k, v in targets.items()},
            {k: v.cuda(non_blocking=True) for k, v in masks.items()})


def _masked_mean(err, mask):
    """Mean of err over samples where mask is True; denom clamped so a task absent
    from the whole batch contributes 0 (not NaN)."""
    m = mask.to(err.dtype)
    return (err * m).sum() / m.sum().clamp_min(1.0)


def _mt_loss(out, targets, masks, stats, weights, seg):
    """Masked multitask loss (std-normalized, weighted) + per-task PHYSICAL MAE.
    Per-atom tasks (forces/magmom) broadcast their per-structure mask over atoms via
    seg. A task is only scored when both the prediction and its target are present."""
    losses, maes = {}, {}

    def scalar(key, err):                          # per-structure (B,) error
        s, m = stats.get(key, 1.0), masks[key]
        losses[key] = _masked_mean(err / s, m)
        maes[key] = _masked_mean(err.detach(), m)

    def per_atom(key, err):                        # per-atom (N,) error, per-structure mask
        s, m = stats.get(key, 1.0), masks[key][seg]
        losses[key] = _masked_mean(err / s, m)
        maes[key] = _masked_mean(err.detach(), m)

    if "energy" in out and "energy" in targets:
        scalar("energy", (out["energy"] - targets["energy"]).abs())
    if "forces" in out and "forces" in targets:
        per_atom("forces", (out["forces"] - targets["forces"]).abs().sum(-1))
    if "stress" in out and "stress" in targets:
        scalar("stress", (out["stress"] - targets["stress"]).abs().flatten(1).mean(1))
    if "magmom" in out and "magmom" in targets:
        per_atom("magmom", (out["magmom"] - targets["magmom"]).abs().sum(-1))
    if "bandgap" in out and "bandgap" in targets:
        scalar("bandgap", (out["bandgap"] - targets["bandgap"]).abs())
    if "eph_lambda" in out and "eph_lambda" in targets:
        scalar("eph_lambda", (out["eph_lambda"] - targets["eph_lambda"]).abs())
    if "eph_wlog" in out and "eph_wlog" in targets:
        scalar("eph_wlog", (out["eph_wlog"] - targets["eph_wlog"]).abs())
    if "dos" in out and "dos" in targets:
        scalar("dos", (out["dos"] - targets["dos"]).abs().mean(1))
    if "eph_a2f" in out and "eph_a2f" in targets:
        scalar("eph_a2f", (out["eph_a2f"] - targets["eph_a2f"]).abs().mean(1))
    if "ph_dos" in out and "ph_dos" in targets:
        scalar("ph_dos", (out["ph_dos"] - targets["ph_dos"]).abs().mean(1))

    if not losses:
        raise ValueError("multitask loss has no terms: the model's tasks and the batch's "
                         "target keys don't overlap (e.g. a 'dos' task with no dos target "
                         "in the collate). Align the config's tasks with the available targets.")
    total = sum(weights.get(k, 1.0) * v for k, v in losses.items())
    return total, maes


def _train_mt(loader, model, optimizer, epoch, stats, weights, args,
              dist_info=None, max_steps=None):
    model.train()
    loss_meter, maes = AverageMeter(), {}
    # Under data parallelism every rank must run the SAME number of optimizer steps
    # (an unequal count deadlocks the next collective), so the schedule uses the
    # rank-synced `max_steps` (the min batch count across shards) and we break there.
    steps = max_steps if max_steps is not None else len(loader)
    warmup_steps = _resolve_warmup_steps(args, steps)
    base_lr = args["learning_rate"]
    is_main = dist_info is None or dist_info.is_main
    t0 = time.time(); data_t = 0.0; end = time.time()
    for i, (input_batch, targets, masks, _cids) in enumerate(loader):
        if i >= steps:                             # drop the tail past the synced count
            break
        data_t += time.time() - end
        input_var = _to_input_var(input_batch, args["cuda"])
        cart, strain = _build_cart_strain(input_var)
        out = model(*input_var, cart=cart, strain=strain)
        targets, masks = _move_target_dicts(targets, masks, args["cuda"])
        loss, batch_maes = _mt_loss(out, targets, masks, stats, weights, input_var[6])
        # Per-step guard: double-backward through 1/|d| reciprocals is more
        # explosion-prone than single-backward, and an NaN here would step + be
        # checkpointed before the per-epoch guard fires. Under data parallelism the
        # check must be COLLECTIVE: a NaN on one rank's shard has to abort ALL ranks,
        # else the finite ranks block forever at the all-reduce below (the min over
        # int(finite) is 0 iff any rank saw a non-finite loss). Single-process ->
        # all_reduce_min_int returns the local value, byte-identical to before.
        finite = int(bool(torch.isfinite(loss)))
        if dist_info is not None and dist_info.enabled:
            finite = all_reduce_min_int(finite, dist_info)
        if not finite:
            if is_main:
                print(f"Exit: non-finite multitask loss at epoch {epoch} step {i} "
                      "(lower learning_rate / raise grad_clip).")
            sys.exit(1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                            # DOUBLE backward (force/stress terms)
        # Average gradients across ranks BEFORE clip/step so every replica steps
        # identically (our explicit stand-in for DDP; no-op single-process).
        if dist_info is not None:
            all_reduce_grads(model, dist_info)
        _warmup_and_step(optimizer, model, args, base_lr, warmup_steps, epoch * steps + i)
        loss_meter.update(float(loss.detach()), 1)
        for k, v in batch_maes.items():
            maes.setdefault(k, AverageMeter()).update(float(v.detach()), 1)
        end = time.time()
        if is_main and i % args.get("print_split", 10) == 0:
            tstr = "  ".join(f"{k} {m.avg:.4f}" for k, m in maes.items())
            print(f"Epoch [{epoch}][{i}/{steps}]  loss {loss_meter.avg:.4f}  {tstr}")
    return loss_meter.avg, {k: m.avg for k, m in maes.items()}, data_t, time.time() - t0


def _validate_mt(loader, model, stats, weights, args):
    model.eval()
    loss_meter, maes = AverageMeter(), {}
    for input_batch, targets, masks, _cids in loader:
        input_var = _to_input_var(input_batch, args["cuda"])
        cart, strain = _build_cart_strain(input_var)
        with torch.enable_grad():                  # forces need a graph even in eval
            out = model(*input_var, cart=cart, strain=strain)
        targets, masks = _move_target_dicts(targets, masks, args["cuda"])
        loss, batch_maes = _mt_loss(out, targets, masks, stats, weights, input_var[6])
        loss_meter.update(float(loss.detach()), 1)
        for k, v in batch_maes.items():
            maes.setdefault(k, AverageMeter()).update(float(v.detach()), 1)
    return loss_meter.avg, {k: m.avg for k, m in maes.items()}


def run_multitask(args, model, optimizer, scheduler, loaders, stats, weights,
                  dist_info=None):
    """Multitask pretraining loop. Checkpoints on the val ENERGY MAE (the primary
    transferable signal); the real evaluation is the downstream T_c transfer, not a
    held-out pretraining metric, so there's no test-CSV / SWA / balanced-val here.

    Data-parallel (dist_info.enabled): each rank trains on its own data shard with
    gradients all-reduced every step (in _train_mt); only rank 0 validates, logs, and
    checkpoints (the replicas are kept bit-identical by the grad all-reduce, so rank
    0's weights represent all). Single-process when dist_info is None/disabled —
    byte-identical to before."""
    _setup_matmul_precision(args)
    train_loader, val_loader = loaders["train"], loaders["val_realistic"]
    shard_sampler = loaders.get("train_shard_sampler")
    is_main = dist_info is None or dist_info.is_main
    best = float("inf")
    monitor = log = w = None
    if is_main:
        from resmon import ResourceMonitor
        monitor = ResourceMonitor(interval=2.0).start()
        print(ResourceMonitor.describe())
    task_cols = [k for k in ("energy", "forces", "stress", "magmom", "bandgap", "dos")
                 if k in stats]
    if is_main:
        print(f"Multitask targets (with data): {task_cols}; loss weights: "
              f"{{ {', '.join(f'{k}:{weights.get(k, 1.0)}' for k in task_cols)} }}"
              + (f"  [data-parallel x{dist_info.world_size}]" if dist_info and dist_info.enabled else ""))
        log = open(args["out_file"] + "_epoch_log.csv", "w", newline="")
        w = csv.writer(log)
        w.writerow(["epoch", "train_loss", "val_loss"]
                   + [f"train_{k}_mae" for k in task_cols]
                   + [f"val_{k}_mae" for k in task_cols]
                   + ["lr", "epoch_time_sec", "train_time_sec", "data_time_sec",
                      "gpu_util_pct", "gpu_mem_gb", "is_best"])

    for epoch in range(args["epochs"]):
        e0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        if shard_sampler is not None:                 # distributed: reshuffle this rank's
            # shard for the new epoch. set_epoch on the size-grouped sampler drops its
            # cached layout AND forwards the epoch to the shard; with plain batching there's
            # no size-grouped sampler, so the shard is set directly.
            bsamp = getattr(train_loader, "batch_sampler", None)
            if bsamp is not None and hasattr(bsamp, "set_epoch"):
                bsamp.set_epoch(epoch)
            else:
                shard_sampler.set_epoch(epoch)
        # All ranks must run the SAME #steps; the size-grouped sampler gives each shard
        # a different batch count, so sync to the min (len() materializes the cached
        # layout that __iter__ then reuses, so this doesn't re-shuffle).
        max_steps = all_reduce_min_int(len(train_loader), dist_info) if dist_info else None
        if max_steps == 0:                            # a shard too small to form one batch
            # The synced min is identical on every rank, so ALL ranks skip together —
            # no collective inside the skipped region runs on a subset of ranks.
            if is_main:
                print(f">> WARNING: epoch {epoch} has 0 synced steps (a rank's shard yielded "
                      "no batch); skipping this epoch. Reduce world_size or max_atoms_per_batch.")
            continue
        tr_loss, tr_maes, data_s, tr_s = _train_mt(
            train_loader, model, optimizer, epoch, stats, weights, args,
            dist_info=dist_info, max_steps=max_steps)

        # Only rank 0 validates (the replicas are identical; val is cheap and needs no
        # collective). Other ranks skip straight to the barrier below.
        va_loss, va_maes = (_validate_mt(val_loader, model, stats, weights, args)
                            if is_main else (0.0, {}))
        # Collective val-NaN guard: only rank 0 validates, so its verdict must be shared —
        # a rank-0-only sys.exit would strand the other ranks forever at the epoch barrier.
        # (va_loss != va_loss is True only for NaN; non-main ranks vote 1 = finite.)
        va_finite = int(va_loss == va_loss) if is_main else 1
        if dist_info is not None and dist_info.enabled:
            va_finite = all_reduce_min_int(va_finite, dist_info)
        if not va_finite:
            if is_main:
                print("Exit due to NaN")
            sys.exit(1)
        scheduler.step()                              # deterministic, all ranks in step

        if is_main:
            val_metric = va_maes.get("energy", va_loss)   # checkpoint on energy MAE
            is_best = val_metric < best
            best = min(val_metric, best)
            _save_checkpoint({
                "epoch": epoch + 1, "state_dict": model.state_dict(), "best": best,
                "optimizer": optimizer.state_dict(), "target_stats": stats,
                "loss_weights": weights, "args": args,
            }, is_best, args["out_file"])
            res = monitor.epoch_stats()
            w.writerow([epoch, float(tr_loss), float(va_loss)]
                       + [tr_maes.get(k, "") for k in task_cols]
                       + [va_maes.get(k, "") for k in task_cols]
                       + [lr, time.time() - e0, round(tr_s, 2), round(data_s, 2),
                          res["gpu_util_pct"], res["gpu_mem_gb"], int(is_best)])
            log.flush()
            tstr = "  ".join(f"{k}={va_maes[k]:.4f}" for k in task_cols if k in va_maes)
            print(f">> epoch {epoch}: train_loss {tr_loss:.4f}  val_loss {va_loss:.4f}  "
                  f"val[{tstr}]  ({time.time() - e0:.1f}s, data-wait {data_s:.1f}s)")
        # Keep ranks aligned before the next epoch's collective (rank 0 spent time in
        # validation/checkpoint while the others idled here).
        barrier(dist_info) if dist_info else None

    if is_main:
        monitor.stop()
        log.close()
        print(f"Multitask pretraining done. Best val energy MAE: {best:.4f}")
