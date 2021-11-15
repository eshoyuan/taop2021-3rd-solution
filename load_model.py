from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import timm
from timm.data.transforms_factory import create_transform
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets.get_dataset import get_dataset_jpg_ti, get_dataset_jpg_ti_onehot, get_predictation_dataset_jpg_ti_label
from utils import Bar, Logger, AverageMeter, odir_metrics, mkdir_p, savefig, LabelSmoothSoftmaxCEV1
from utils.eval import ti_metrics_single_class
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib.pyplot as plt
import csv
from pytorch_metric_learning import miners, reducers, testers
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from datasets.dataloader import Ti_jpg, Ti_jpg_onehot, Ti_jpg_predict
def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_m = AverageMeter()
    acc = np.zeros(args.num_labels)
    score_acc = AverageMeter()
    end = time.time()
    # metrci learning
    miner = miners.MultiSimilarityMiner()
    loss_func = TripletMarginLoss()
    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        targets_list = targets.tolist()

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        if args.arch == 'inception_v3':
            outputs, aux = model(inputs)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(aux, targets)
            loss = loss1 + 0*loss2
        else:
            outputs = model(inputs)
            hard_pairs = miner(outputs, targets)
            loss2 = loss_func(outputs, targets, hard_pairs)

            loss1 = criterion(outputs, targets)
            loss = loss1

        acc = compute_acc(outputs, targets_list)
        score_acc.update(acc, inputs.size(0))
        losses_m.update(loss2.data, inputs.size(0))
        losses.update(loss1.data, inputs.size(0))

        # compute gradient and do AdamW step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_m: {loss2:.4f} | ACC: {score_acc: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss2=losses_m.avg,
            score_acc=score_acc.avg,
        )
        bar.next()
    bar.finish()

    # acc = acc / n_train
    # print(acc)
    return (losses.avg, score_acc.avg)


def test(testloader, model, criterion, use_cuda):
    global best_acc

    miner = miners.MultiSimilarityMiner()
    loss_func = TripletMarginLoss()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = np.zeros(args.num_labels)
    score_acc = AverageMeter()
    losses_m = AverageMeter()

    # switch to evaluate mode
    # model.eval()
    if args.eval == False:
        # switch to train mode
        model.train()
    else:
        model.eval()
    """
    Here, if use model.eval(), it will severely decrease the performance.  
    It is possible that your training in general is unstable, so BatchNorm’s running_mean and running_var dont represent true batch statistics.
    http://pytorch.org/docs/master/nn.html?highlight=batchnorm#torch.nn.BatchNorm1d 422
    Try the following:
    1. change the momentum term in BatchNorm constructor to higher.
    2. before you set model.eval(), run a few inputs through model (just forward pass, you dont need to backward). 
       This will help stabilize the running_mean / running_std values.
    """

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)
            targets_list = targets.tolist()
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # compute output
            if args.arch == 'inception_v3':
                outputs, aux = model(inputs)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(aux, targets)
                loss = loss1 + loss2
            else:
                outputs = model(inputs)
                hard_pairs = miner(outputs, targets)
                loss2 = loss_func(outputs, targets, hard_pairs)

                loss1 = criterion(outputs, targets)
                #loss = loss1+ 0.3 *loss2

            acc = compute_acc(outputs, targets_list)
            score_acc.update(acc, inputs.size(0))
            losses_m.update(loss2.data, inputs.size(0))
            losses.update(loss1.data, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_m: {loss2:.4f} | ACC: {score_acc: .4f}'.format(
                batch=batch_idx + 1,
                size=len(testloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss2=losses_m.avg,
                score_acc=score_acc.avg,
            )
            bar.next()
        bar.finish()
    return (losses.avg, score_acc.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch==0:
        state['lr'] =args.lr
    # if epoch<3:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 1e-5
    # if epoch==3:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def single_test_TTA_hard(model, testloader, use_cuda, epoch, savepath, num_TTA=8):
    """TTA+硬投票, num_TTA为进行TTA的次数"""
    global best_acc

    # switch to evaluate mode
    # model.eval()
    if args.eval == False:
        # switch to train mode
        model.train()
    else:
        model.eval()

    # model.to(device)

    res = []
    targets_list = []
    all_targets = torch.tensor(2.5)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            all_targets = targets.clone()
        else:
            all_targets = torch.cat((all_targets, targets), 0)
    # print(all_targets.shape)

    for i in range(num_TTA):
        res.append([])
    with torch.no_grad():
        for n in range(num_TTA):  # TTA的次数
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                if n == 0:
                    if args.format_label == 'scalar':
                        targets_list += targets.tolist()
                    if args.format_label == 'onehot':
                        targets_list += targets.argmax(dim=1).tolist()
                outs = model(inputs)
                for i in range(outs.shape[0]):
                    res[n].append(outs.argmax(dim=1)[i].item())
    vote_res = []

    for i in range(len(res[0])):
        temp = []
        for j in range(num_TTA):
            temp.append(res[j][i])
        vote_res.append(np.argmax(np.bincount(temp)).item())
    # cm = confusion_matrix(targets_list, vote_res, labels=[0, 1, 2, 3, 4])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    # disp.plot()
    # plt.savefig(savepath + f'/confusion_matrix_{epoch}.png')
    # predicts_onehot = convert_label_to_onehot(vote_res)
    # print(predicts_onehot.shape)
    acc = (np.array(vote_res) == np.array(
        targets_list)).sum() / len(targets_list)

    print("TTA_acc = ", acc)
    return use_cuda, acc


def single_test_TTA_soft(model, testloader, use_cuda, epoch, savepath, num_TTA=8):
    """TTA+软投票, num_TTA为进行TTA的次数"""
    global best_acc

    if args.eval == False:
        model.train()
    else:
        model.eval()

    res = []
    targets_list = []
    all_targets = torch.tensor(2.5)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            all_targets = targets.clone()
        else:
            all_targets = torch.cat((all_targets, targets), 0)

    for i in range(num_TTA):
        res.append([])

    with torch.no_grad():
        totaloutputs = None
        for n in range(num_TTA):  # TTA的次数
            alloutputs = None
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                if n == 0:
                    if args.format_label == 'scalar':
                        targets_list += targets.tolist()
                    if args.format_label == 'onehot':
                        targets_list += targets.argmax(dim=1).tolist()
                outs = model(inputs)
                if alloutputs == None:
                    alloutputs = outs
                else:
                    alloutputs = torch.cat((alloutputs, outs), 0)
            if totaloutputs == None:
                totaloutputs = alloutputs
            else:
                totaloutputs += alloutputs
    totaloutputs /= num_TTA
    acc = compute_acc(totaloutputs, targets_list)
    print("TTA_acc = ", acc)
    return use_cuda, acc
def single_test_TTA_hard_pred(model, testloader, use_cuda, num_TTA=8, eval=False):
    global best_acc
    # switch to train mode
    if eval != None:
        if eval == False:
            model.train()
        else:
            model.eval()
        # model.to(device)

    res = []
    target = []
    all_targets = torch.tensor(2.5)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            all_targets = targets.clone()
        else:
            all_targets = torch.cat((all_targets, targets), 0)
    # print(all_targets.shape)

    for i in range(num_TTA):
        res.append([])
    with torch.no_grad():
        for n in range(num_TTA):  # TTA的次数
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                if n == 0:
                    target += targets.tolist()
                imgs = inputs
                outs = model(imgs)
                # data[1]中存储的数据是label
                for i in range(outs.shape[0]):
                    res[n].append(outs.argmax(dim=1)[i].item())

    vote_res = []
    for i in range(len(res[0])):
        temp = []
        for j in range(num_TTA):
            temp.append(res[j][i])
        vote_res.append(np.argmax(np.bincount(temp)).item())

    # predicts_onehot = convert_label_to_onehot(vote_res)
    return vote_res
# 决定backbone model使用的loss function


def decide_model_criterion():
    criterion = None
    if args.criterion.lower() == 'bcewithlogitsloss':
        # 若使用BCEWithLogitsLoss作为loss function，那么labels应为onehot的形式
        args.format_label = 'onehot'

        #     use BCEWithLogitsLoss as criterion with one-hot
        if args.weighted_CE == 1:
            criterion = nn.BCEWithLogitsLoss(weight=class_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
            print('weights = [1 1 1 1 1]')

    elif args.criterion.lower() == 'crossentropyloss':
        # 若使用CrossEntropyLoss作为loss function，那么labels应为scalar的形式
        args.format_label = 'scalar'

        # use CrossEntropyLoss as criterion with non one-hot labels
        if args.weighted_CE == 1:
            criterion = nn.CrossEntropyLoss(weight=class_weights.cuda())
        else:
            criterion = nn.CrossEntropyLoss()
            print('weights = [1 1 1 1 1]')

    return criterion
def predict(res,path):
    # switch to train mode
    file = 'taop-2021/100001/To user/test2_data_info.csv'
    data = pd.read_csv(file, encoding='utf-8')
    for i in range(len(data)):
        data[u'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'][i] = str(int(res[i]+1))
    data[u'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'] = data[u'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'].astype(int)
    data.to_csv(path, index=False, encoding='utf-8')
# 由给定activate function计算acc


def compute_acc(outputs, targets_list):
    # 其实这里不需要区分激活函数, softmax和sigmoid都是单调函数, 取outputs最大即可
    acc = None
    if args.activate.lower() == 'softmax':
        if args.format_label == 'scalar':
            pred = torch.softmax(outputs, 1).detach().cpu().numpy()
            res = []
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            acc = (np.array(res) == np.array(
                targets_list)).sum() / len(targets_list)
        if args.format_label == 'onehot':
            pred = torch.softmax(outputs, 1).detach().cpu().numpy()
            res = []
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            # 若target的形式为onehot，则转换为scalar
            targets_list_scalar = []
            for i in range(len(targets_list)):
                targets_list_scalar.append(np.argmax(targets_list[i]))
            acc = (np.array(res) == np.array(
                targets_list_scalar)).sum() / len(targets_list)

    elif args.activate.lower() == 'sigmoid':
        if args.format_label == 'scalar':
            pred = torch.sigmoid(outputs).detach().cpu().numpy()
            res = []
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            acc = (np.array(res) == np.array(
                targets_list)).sum() / len(targets_list)
        if args.format_label == 'onehot':
            pred = torch.sigmoid(outputs).detach().cpu().numpy()
            res = []
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            # 若target的形式为onehot，则转换为scalar
            targets_list_scalar = []
            for i in range(len(targets_list)):
                targets_list_scalar.append(np.argmax(targets_list[i]))
            acc = (np.array(res) == np.array(
                targets_list_scalar)).sum() / len(targets_list)
    return acc
model_path="final/seresnext50.pth"
input_size = 448

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
dataset_path = "png"


model = timm.create_model("gluon_seresnext50_32x4d", pretrained=True, num_classes=5)
model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
# Resume      

model.load_state_dict(torch.load(model_path))
testset = get_predictation_dataset_jpg_ti_label(size=input_size)
testloader = data.DataLoader(
    testset, batch_size=32, shuffle=False)
res = single_test_TTA_hard_pred(model, testloader, 1, num_TTA=8)
predict(res,"./results.csv")
