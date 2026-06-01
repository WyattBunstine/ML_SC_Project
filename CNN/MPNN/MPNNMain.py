import csv
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

from MPNNData import (CIFDataV4, collate_pool, get_train_val_test_loader,
                      get_classification_loaders, NODE_FEA_LEN, NBR_FEA_LEN)
from MPNNModel import CrystalMPNN

best_mae_error = 1e10


def main():
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        warnings.warn("Usage: MPNNMain.py <config.json>")
        return -1

    with open(sys.argv[1]) as f:
        args = json.load(f)

    classification = args.get("task", "regression") == "classification"

    dataset = CIFDataV4(
        index_path=args["index_path"],
        max_num_nbr=args.get("max_num_nbr", 14),
    )

    normalizer = None
    loaders = None
    if classification:
        loaders = get_classification_loaders(
            dataset,
            batch_size=args["batch_size"],
            val_ratio=args["val_ratio"],
            test_ratio=args["test_ratio"],
            n_nonsc=args.get("n_nonsc", 5000),
            num_workers=args.get("num_workers", 0),
            pin_memory=torch.cuda.is_available(),
            seed=args.get("split_seed", 123),
        )
        print("Classification split sizes:", loaders["split_sizes"])
        train_loader = loaders["train"]
    else:
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_pool,
            batch_size=args["batch_size"],
            val_ratio=args["val_ratio"],
            test_ratio=args["test_ratio"],
            num_workers=args.get("num_workers", 0),
            train_size=None, test_size=None, val_size=None,
            pin_memory=torch.cuda.is_available(),
            return_test=True,
        )
        # Normalizer for target (T_c)
        sample_targets = torch.cat([t for _, t, _lab, _ in train_loader])
        normalizer = Normalizer(sample_targets)

    # Infer feature dims from first sample
    (sample_atom, sample_nbr, _), _, _lab, _ = dataset[0]
    orig_atom_fea_len = sample_atom.shape[-1]
    nbr_fea_len = sample_nbr.shape[-1]

    model = CrystalMPNN(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=args.get("atom_feat_len", 64),
        edge_hidden_dim=args.get("edge_hidden_dim", 128),
        n_conv=args.get("n_conv", 3),
        h_fea_len=args.get("h_feat_len", 128),
        n_h=args.get("n_hidden", 1),
        aggregation=args.get("aggregation", "ecn_weighted"),
        classification=classification,
    )

    args["cuda"] = torch.cuda.is_available()
    if args["cuda"]:
        model.cuda()

    criterion = nn.NLLLoss() if classification else nn.L1Loss()

    if args.get("optim", "SGD") == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args["learning_rate"],
                              momentum=args.get("momentum", 0.9),
                              weight_decay=args.get("weight_decay", 0))
    elif args["optim"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"],
                               weight_decay=args.get("weight_decay", 0))
    else:
        raise ValueError(f"Unsupported optim: {args['optim']}")

    scheduler = MultiStepLR(optimizer, milestones=args.get("lr_milestones", [100]), gamma=0.1)

    if classification:
        _run_classification(args, model, criterion, optimizer, scheduler, loaders)
    else:
        _run_regression(args, model, criterion, optimizer, scheduler,
                        train_loader, val_loader, test_loader, normalizer)


def _run_regression(args, model, criterion, optimizer, scheduler,
                    train_loader, val_loader, test_loader, normalizer):
    global best_mae_error
    train_losses, val_losses = [], []

    # per-epoch telemetry log (mirrors CGCNNMain.py)
    epoch_log_file = open(args["out_file"] + "_epoch_log.csv", "w", newline="")
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(["epoch", "train_loss", "train_mae", "val_loss",
                           "val_mae", "lr", "epoch_time_sec", "is_best"])

    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]["lr"]

        train_loss, train_mae = _train(train_loader, model, criterion, optimizer, epoch, normalizer, args)
        mae_error, val_loss = _validate(val_loader, model, criterion, normalizer, args)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if mae_error != mae_error:
            print("Exit due to NaN")
            sys.exit(1)

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
                               float(val_loss), float(mae_error), lr,
                               epoch_time, int(is_best)])
        epoch_log_file.flush()

    epoch_log_file.close()

    np.save(args["out_file"] + "_losstrain.csv", np.array(train_losses))
    np.save(args["out_file"] + "_lossval.csv", np.array(val_losses))

    print("-" * 50 + "\nEvaluating on test set")
    best_ckpt = torch.load(args["out_file"] + "_model_best.pth.tar")
    model.load_state_dict(best_ckpt["state_dict"])
    _validate(test_loader, model, criterion, normalizer, args, test=True)


def _run_classification(args, model, criterion, optimizer, scheduler, loaders):
    """Train + evaluate the MPNN as a Stage-1 SC/non-SC classifier.

    Model selection uses validation AUC on the realistic (imbalanced) split;
    metrics logged on both realistic and balanced val; test evaluated on both.
    """
    best_auc = -1.0
    train_loader = loaders["train"]
    val_real = loaders["val_realistic"]
    val_bal = loaders["val_balanced"]

    epoch_log_file = open(args["out_file"] + "_epoch_log.csv", "w", newline="")
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(["epoch", "train_loss", "train_acc", "val_loss",
                           "val_acc", "val_precision", "val_recall", "val_f1",
                           "val_auc", "val_bal_acc", "lr", "epoch_time_sec", "is_best"])

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
        select_metric = val_auc if val_auc == val_auc else val_metrics["f1"]
        is_best = select_metric > best_auc
        best_auc = max(select_metric, best_auc)
        _save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_auc": best_auc,
            "optimizer": optimizer.state_dict(),
            "args": args,
        }, is_best, args["out_file"])

        epoch_time = time.time() - epoch_start
        epoch_logger.writerow([epoch, float(train_loss), float(train_acc),
                               float(val_loss), float(val_metrics["accuracy"]),
                               float(val_metrics["precision"]), float(val_metrics["recall"]),
                               float(val_metrics["f1"]), float(val_auc),
                               float(bal_metrics["accuracy"]), lr, epoch_time, int(is_best)])
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


def _validate(loader, model, criterion, normalizer, args, test=False):
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
        results_path = args["out_file"] + ".csv"
        with open(results_path, "w", newline="") as f:
            writer = csv.writer(f)
            for cif_id, tgt, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, tgt, pred))

    return mae_errors.avg, losses.avg


def _to_input_var(input_batch, cuda):
    """Move a collated input tuple onto the right device."""
    if cuda:
        return (
            Variable(input_batch[0].cuda(non_blocking=True)),
            Variable(input_batch[1].cuda(non_blocking=True)),
            input_batch[2].cuda(non_blocking=True),
            [idx.cuda(non_blocking=True) for idx in input_batch[3]],
        )
    return (Variable(input_batch[0]), Variable(input_batch[1]),
            input_batch[2], input_batch[3])


def classification_metrics(log_probs, targets):
    """Metrics for the SC/non-SC head. log_probs: (N,2) log-softmax CPU tensor;
    targets: (N,) long tensor. AUC is nan if the set is single-class."""
    probs = np.exp(log_probs.numpy())
    pred_label = np.argmax(probs, axis=1)
    target_label = targets.numpy().reshape(-1)
    accuracy = metrics.accuracy_score(target_label, pred_label)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target_label, pred_label, average="binary", zero_division=0)
    try:
        auc = metrics.roc_auc_score(target_label, probs[:, 1])
    except ValueError:
        auc = float("nan")
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": fscore, "auc": auc}


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
    m = classification_metrics(log_probs, labels)

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
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


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
