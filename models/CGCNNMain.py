# most of this code is taken from https://github.com/txie-93/cgcnn with some modification for the project

import argparse
import csv
import datetime
import os
import shutil
import sys
import time
import json
import warnings
from random import sample

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from OriginalCGCNN.data import CIFData
from OriginalCGCNN.data import collate_pool, get_train_val_test_loader, get_classification_loaders
from OriginalCGCNN.CGCNNOrig import CrystalGraphConvNet

best_mae_error = 1e10


def _write_run_metadata(run_dir, args, model, train_loader, dataset, classification,
                        feature_dims):
    """Write run_dir/metadata.json: resolved hyperparameters + model-size stats
    for the baseline CGCNN. Mirrors the MPNN trainer's metadata."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        n_train = len(train_loader.sampler)        # SubsetRandomSampler / balanced sampler
    except TypeError:
        n_train = len(train_loader.dataset)
    node_dim, edge_dim = feature_dims

    meta = {
        "run_id": args.get("run_id"),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "model_type": args.get("run_tag", "Orig"),
        "task": "classification" if classification else "regression",
        "config_file": os.path.abspath(sys.argv[1]),
        "feature_dims": {"node": int(node_dim), "edge": int(edge_dim)},
        "architecture": {
            "atom_feat_len": args["atom_feat_len"],
            "n_conv": args["n_conv"],
            "h_feat_len": args["h_feat_len"],
            "n_hidden": args["n_hidden"],
        },
        "training": {
            "optim": args["optim"],
            "learning_rate": args["learning_rate"],
            "weight_decay": args.get("weight_decay", 0),
            "momentum": args.get("momentum"),
            "lr_milestones": args["lr_milestones"],
            "epochs": args["epochs"],
            "batch_size": args["batch_size"],
        },
        "dataset": {
            "dataset": args.get("dataset"),
            "total_indexed": len(dataset),
        },
        "model_size": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "effective_train_samples_per_epoch": n_train,
            "params_per_train_sample": round(total_params / max(1, n_train), 2),
        },
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model: {total_params:,} params "
          f"({meta['model_size']['params_per_train_sample']} per train sample, "
          f"{n_train} effective train samples/epoch)")


def main():
    args = {}
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1]) as f:
            args = json.load(f)
    else:
        warnings.warn("config file not specified")
        return -1

    classification = args.get("task", "regression") == "classification"

    # --- Per-run output directory: model_data/<date>/<run_tag>/<run_id>/ ---
    # Mirrors the MPNN trainer: each run is self-contained (config copy +
    # metadata.json + all artifacts), nested under <date>/<run_tag>/ to keep
    # model_data/ navigable. run_tag defaults to "Orig" for the baseline.
    run_tag = args.get("run_tag", "Orig")
    model_data_dir = args.get("model_data_dir", "model_data")
    now = datetime.datetime.now()
    run_id = f"{run_tag}_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    run_dir = os.path.join(model_data_dir, now.strftime('%Y-%m-%d'), run_tag, run_id)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(sys.argv[1], os.path.join(run_dir, "config.json"))
    out_base = os.path.basename(args.get("out_file", "result")) or "result"
    args["out_file"] = os.path.join(run_dir, out_base)
    args["run_id"] = run_id
    print(f"Run output dir: {run_dir}")

    # load data (OriginalCGCNN baseline; the CE variant has been retired).
    # target_column selects the regression target for multi-target pickles (e.g.
    # the MP energy dataset: "formation_energy_per_atom" / "e_above_hull"); unset
    # falls back to the legacy `value`/`tc` column, so SC configs are unaffected.
    dataset = CIFData(args["dataset_rd"], args["atom_init"], args["dataset"],
                      target_column=args.get("target_column"))

    normalizer = None
    loaders = None
    if classification:
        # Stage-1 SC/non-SC classifier: stratified split + per-epoch balanced sampler.
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
        val_loader = test_loader = None
    else:
        train_loader, val_loader, test_loader = get_train_val_test_loader(
            dataset=dataset,
            collate_fn=collate_pool,
            batch_size=args["batch_size"],
            train_ratio=None,
            val_ratio=args["val_ratio"],
            test_ratio=args["test_ratio"],
            num_workers=args.get("num_workers", 0),
            train_size=None,
            test_size=None,
            val_size=None,
            pin_memory=torch.cuda.is_available(),
            return_test=True)

        sample_target = [target for i, (input, target, _label, _) in enumerate(train_loader)]
        sample_target = torch.cat(sample_target)
        normalizer = Normalizer(torch.Tensor(sample_target))

    # build model
    structures, _, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args["atom_feat_len"],
                                n_conv=args["n_conv"],
                                h_fea_len=args["h_feat_len"],
                                n_h=args["n_hidden"],
                                classification=classification)
    if torch.cuda.is_available():
        model.cuda()
        args["cuda"] = True
    else:
        args["cuda"] = False

    # Record run metadata (hyperparameters + model size) before training.
    _write_run_metadata(run_dir, args, model, train_loader, dataset, classification,
                        feature_dims=(orig_atom_fea_len, nbr_fea_len))

    criterion = nn.NLLLoss() if classification else nn.L1Loss()
    if args["optim"] == 'SGD':
        optimizer = optim.SGD(model.parameters(), args["learning_rate"],
                              momentum=args["momentum"],
                              weight_decay=args["weight_decay"])
    elif args["optim"] == 'Adam':
        optimizer = optim.Adam(model.parameters(), args["learning_rate"],
                               weight_decay=args["weight_decay"])
    else:
        raise NameError('Only SGD or Adam is allowed as --optim')

    scheduler = MultiStepLR(optimizer, milestones=args["lr_milestones"],
                            gamma=0.1)

    if classification:
        run_classification(args, model, criterion, optimizer, scheduler, loaders)
    else:
        run_regression(args, model, criterion, optimizer, scheduler,
                       train_loader, val_loader, test_loader, normalizer)


def run_regression(args, model, criterion, optimizer, scheduler,
                   train_loader, val_loader, test_loader, normalizer):
    global best_mae_error

    train_losses = [];
    val_losses = [];

    # per-epoch telemetry log
    epoch_log_path = args["out_file"] + '_epoch_log.csv'
    epoch_log_file = open(epoch_log_path, 'w', newline='')
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(['epoch', 'train_loss', 'train_mae', 'val_loss',
                           'val_mae', 'lr', 'epoch_time_sec', 'is_best'])

    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']

        # train for one epoch
        train_loss, train_mae = train(train_loader, model, criterion, optimizer, epoch, normalizer, args)

        # evaluate on validation set
        mae_error, val_loss = validate(val_loader, model, criterion, normalizer, args)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if mae_error != mae_error:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
        }, is_best, args["out_file"])

        epoch_time = time.time() - epoch_start
        epoch_logger.writerow([epoch, float(train_loss), float(train_mae),
                               float(val_loss), float(mae_error), lr,
                               epoch_time, int(is_best)])
        epoch_log_file.flush()

    epoch_log_file.close()

    train_losses = np.array(train_losses);
    val_losses = np.array(val_losses)
    loss_filename = args["out_file"] + '_loss'
    np.save(loss_filename + 'train.csv', train_losses);
    np.save(loss_filename + 'val.csv', val_losses)

    # test best model
    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load(args["out_file"] + '_model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, args, test=True)


def run_classification(args, model, criterion, optimizer, scheduler, loaders):
    """Train + evaluate the Stage-1 SC/non-SC classifier.

    Model selection (is_best) uses validation AUC on the *realistic* (imbalanced)
    split. Metrics are logged on both the realistic and balanced val splits, and
    the test set is evaluated on both at the end.
    """
    best_auc = -1.0
    train_loader = loaders["train"]
    val_real = loaders["val_realistic"]
    val_bal = loaders["val_balanced"]

    epoch_log_path = args["out_file"] + '_epoch_log.csv'
    epoch_log_file = open(epoch_log_path, 'w', newline='')
    epoch_logger = csv.writer(epoch_log_file)
    epoch_logger.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss',
                           'val_acc', 'val_precision', 'val_recall', 'val_f1',
                           'val_auc', 'val_bal_acc', 'lr', 'epoch_time_sec', 'is_best'])

    for epoch in range(args["epochs"]):
        epoch_start = time.time()
        lr = optimizer.param_groups[0]['lr']

        train_loss, train_acc = train_classification(
            train_loader, model, criterion, optimizer, epoch, args)
        val_metrics, val_loss = validate_classification(val_real, model, criterion, args)
        bal_metrics, _ = validate_classification(val_bal, model, criterion, args)

        if val_loss != val_loss:
            print('Exit due to NaN')
            sys.exit(1)

        scheduler.step()

        val_auc = val_metrics["auc"]
        # AUC can be nan if a split happens to be single-class; fall back to F1.
        select_metric = val_auc if val_auc == val_auc else val_metrics["f1"]
        is_best = select_metric > best_auc
        best_auc = max(select_metric, best_auc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_auc': best_auc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args["out_file"])

        epoch_time = time.time() - epoch_start
        epoch_logger.writerow([epoch, float(train_loss), float(train_acc),
                               float(val_loss), float(val_metrics["accuracy"]),
                               float(val_metrics["precision"]), float(val_metrics["recall"]),
                               float(val_metrics["f1"]), float(val_auc),
                               float(bal_metrics["accuracy"]), lr, epoch_time, int(is_best)])
        epoch_log_file.flush()

    epoch_log_file.close()

    print('--------- Evaluate Classifier on Test Set ---------')
    best_checkpoint = torch.load(args["out_file"] + '_model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    test_real, _ = validate_classification(loaders["test_realistic"], model,
                                           criterion, args, test=True, tag="realistic")
    test_bal, _ = validate_classification(loaders["test_balanced"], model,
                                          criterion, args, test=True, tag="balanced")
    print(" ** Test (realistic):", {k: round(v, 4) for k, v in test_real.items()})
    print(" ** Test (balanced): ", {k: round(v, 4) for k, v in test_bal.items()})


def train(train_loader, model, criterion, optimizer, epoch, normalizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, _label, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args["cuda"]:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        # normalize target
        target_normed = normalizer.norm(target)
        if args["cuda"]:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)

        loss = criterion(output, target_var)  # log output, target_var

        # measure accuracy and record loss
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args["print_split"] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mae_errors=mae_errors)
            )

    return losses.avg, mae_errors.avg


def validate(val_loader, model, criterion, normalizer, args, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, _label, batch_cif_ids) in enumerate(val_loader):
        if args["cuda"]:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
        target_normed = normalizer.norm(target)
        if args["cuda"]:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)  # log

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args["print_split"] == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                mae_errors=mae_errors))

    if test:
        star_label = '**'
        import csv
        results_path = args["out_file"] + '.csv'
        with open(results_path, 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = '*'

    print(' {star} MAE {mae_errors.avg:.3f}'.format(star=star_label, mae_errors=mae_errors))
    return mae_errors.avg, losses.avg


def _to_input_var(input, cuda):
    """Move a collated input tuple onto the right device."""
    if cuda:
        return (Variable(input[0].cuda(non_blocking=True)),
                Variable(input[1].cuda(non_blocking=True)),
                input[2].cuda(non_blocking=True),
                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
    return (Variable(input[0]), Variable(input[1]), input[2], input[3])


def classification_metrics(log_probs, targets):
    """Compute classification metrics for the SC/non-SC head.

    log_probs: (N, 2) log-softmax tensor on CPU
    targets:   (N,) long tensor of true labels (1 = SC, 0 = non-SC)
    Returns dict with accuracy, precision, recall, f1, auc (auc is nan if the
    set is single-class).
    """
    probs = np.exp(log_probs.numpy())
    pred_label = np.argmax(probs, axis=1)
    target_label = targets.numpy().reshape(-1)
    pos_prob = probs[:, 1]
    accuracy = metrics.accuracy_score(target_label, pred_label)
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target_label, pred_label, average='binary', zero_division=0)
    try:
        auc = metrics.roc_auc_score(target_label, pos_prob)
    except ValueError:
        auc = float('nan')  # only one class present
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1": fscore, "auc": auc}


def train_classification(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    model.train()
    end = time.time()
    for i, (input, target, label, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_var = _to_input_var(input, args["cuda"])
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

        if i % args["print_split"] == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {bt.val:.3f} ({bt.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                      epoch, i, len(train_loader), bt=batch_time,
                      loss=losses, acc=accuracies))

    return losses.avg, accuracies.avg


def validate_classification(loader, model, criterion, args, test=False, tag=""):
    """Evaluate the classifier over an entire loader; metrics computed once on the
    full set (so AUC is well-defined). When test=True, writes per-sample
    predictions (cif_id, true_label, p_sc) to <out_file>_test_<tag>.csv."""
    losses = AverageMeter()
    model.eval()
    all_log_probs, all_labels, all_cif_ids = [], [], []

    for i, (input, target, label, batch_cif_ids) in enumerate(loader):
        with torch.no_grad():
            input_var = _to_input_var(input, args["cuda"])
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
        results_path = args["out_file"] + ('_test_%s.csv' % tag if tag else '.csv')
        with open(results_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cif_id', 'true_label', 'p_sc'])
            for cid, tl, p in zip(all_cif_ids, labels.numpy().tolist(), pos_prob.tolist()):
                writer.writerow((cid, int(tl), p))

    return m, losses.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:  # should be 2
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average='binary')
        # auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore  # , auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, k, lr):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_checkpoint.pth.tar', filename + '_model_best.pth.tar')



if __name__ == '__main__':
    main()
