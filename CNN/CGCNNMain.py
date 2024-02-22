# most of this code is taken from https://github.com/txie-93/cgcnn with some modification for the project

import argparse
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
from OriginalCGCNN.data import CIFData as OrigCifData
from CGCNNCoordEnv.CEdata import CIFData
from CGCNNCoordEnv.CEdata import collate_pool, get_train_val_test_loader
from CGCNNCoordEnv.CGCNNCE import CrystalGraphConvNet

best_mae_error = 1e10

def main():
    global best_mae_error
    args = {}
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        with open(sys.argv[1]) as f:
            args = json.load(f)
    else:
        warnings.warn("config file not specified")
        return -1
    # load data
    dataset = None
    if "ORIG" in args["models"]:
        dataset = OrigCifData(args["dataset_rd"], args["atom_init"], args["dataset"])
    else:
        dataset = CIFData(args["dataset_rd"], args["atom_init"], args["dataset"])
    # print(dataset[0])
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args["batch_size"],
        train_ratio=None,
        val_ratio=args["val_ratio"],
        test_ratio=args["test_ratio"],
        num_workers=0,
        train_size=None,
        test_size=None,
        val_size=None,
        pin_memory=torch.cuda.is_available(),
        return_test=True)

    sample_target = [target for i, (input, target, _) in enumerate(train_loader)]
    sample_target = torch.cat(sample_target)
    normalizer = Normalizer(torch.Tensor(sample_target))

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,
                                atom_fea_len=args["atom_feat_len"],
                                n_conv=args["n_conv"],
                                h_fea_len=args["h_feat_len"],
                                n_h=args["n_hidden"],
                                classification=False)
    if torch.cuda.is_available():
        model.cuda()
        args["cuda"] = True
    else:
        args["cuda"] = False

    criterion = nn.L1Loss()
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

    train_losses = [];
    val_losses = [];
    for epoch in range(args["epochs"]):
        # train for one epoch
        train_loss = train(train_loader, model, criterion, optimizer, epoch, normalizer, args)

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


def train(train_loader, model, criterion, optimizer, epoch, normalizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
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

    return loss.item()


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
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
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
    return mae_errors.avg, loss.item()


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
