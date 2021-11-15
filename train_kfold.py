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


parser = argparse.ArgumentParser(
    description='PyTorch Single-label Classification Training - Ocular Single-Disease Identification')
parser.add_argument(
    '--dataset', default='./png_crop', type=str, help='dataset path')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--input_size', default=448, type=int,
                    help='the input size of images')
parser.add_argument('--num_labels', default=5, type=int,
                    help='the number of labels')
# Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[10,20,30,50],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.33,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--optim', default="Adam", help="Adam or SGD")
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--weighted_CE', default=0, type=int,
                    help='whether to use weighted cross-entropy loss')

# 指定需要使用的损失函数，目前支持BCE及CE
# 若需使用BCE则参数值为bcewithlogitsloss
# 若需使用CE则参数值为crossentropyloss
parser.add_argument('--criterion', default='crossentropyloss',
                    type=str, help='choose the loss function')

# 这两行可以忽略, 因为会根据loss function自适应
#   指定需要使用的激活函数，目前支持softmax和sigmoid
#   若需使用softmax则参数值为softmax
#   若需使用sigmoid则参数值为sigmoid
parser.add_argument('--activate', default='sigmoid',
                    type=str, help='choose the activation function')
parser.add_argument('--format_label', default='onehot',
                    type=str, help='choose the format of labels')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='whether to load model pretrained on Imagenet')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='repvgg_b3g4')

parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--eval', default=True, type=bool,
                    help='whether to use model.eval()')
parser.add_argument('--num_TTA', default=4)
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
num_labels = args.num_labels
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)


def main():
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    start_time = int(time.time())
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset')
    # Choose backbone model
    # load dataset
    dataset_path = "./png_crop"
    dataset_csv = "taop-2021/100001/To user/train2_data_info.csv"
    image_path_list = []
    image_label_list = []
    with open(dataset_csv) as csvfile:
        csv_reader = csv.reader(csvfile)
        csv_header = next(csv_reader)
        for row in csv_reader:
            image_path = os.path.join(dataset_path, row[2] + '.png')
            image_path_list.append(image_path)
            image_label_list.append(row[7])
    csvfile.close()
    img_paths = []
    labels = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_path_list, image_label_list)):
        img_paths.append(
            ([image_path_list[i] for i in train_idx], [image_path_list[i] for i in val_idx]))
        labels.append(([image_label_list[i] for i in train_idx], [
                      image_label_list[i] for i in val_idx]))
    # Resume
    title = 'DR-GRADING-' + args.arch
    # 创建存储混淆矩阵图片的文件夹
    dirname = f'/{args.arch}_{start_time}'
    os.makedirs(args.checkpoint + str(dirname), mode=0o777)
    logger_all = Logger(
        args.checkpoint + dirname + f'/log_saved.txt',
        title=title,
        resume=False)
    logger_all.set_names(
        ['Fold', 'Best acc', 'epoch'])
    for fold, ((train_img_paths, val_img_paths), (train_labels, val_labels)) in enumerate(zip(img_paths, labels)):
        model = timm.create_model(
            args.arch, pretrained=args.pretrained, num_classes=5)

        # 使得模型能运行在两块显卡上，若使用单卡则需要注释下面这一行
        model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

        cudnn.benchmark = True
        if fold == 0:
            print(f"fold:{fold}")
            print("==> creating model '{}'".format(args.arch))
            print('    Total params: %.2fM' % (sum(p.numel()
                                                   for p in model.parameters()) / 1000000.0))

        # 选择backbone model的loss function
        criterion = decide_model_criterion()
        criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1)
        if args.optim == "Adam":
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=args.lr, momentum=args.momentum)

        # prepare logger
        best_acc = 0.79
        logger = Logger(
            args.checkpoint + dirname + f'/log{fold}_{args.arch}.txt',
            title=title,
            resume=False)
        logger.set_names(
            ['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc', 'TTA Acc'])

        # prepare loader
        trainset = Ti_jpg(image_list=train_img_paths, label=train_labels,
                          phase='train', augmentation=1, size=args.input_size)
        valset = Ti_jpg(image_list=val_img_paths, label=val_labels,
                        phase='val', augmentation=1, size=args.input_size)
        trainloader = data.DataLoader(
            trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        valloader = data.DataLoader(
            valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)

        # begin to train
        for epoch in range(start_epoch, args.epochs):
            adjust_learning_rate(optimizer, epoch)
            print('\nEpoch: [%d | %d] LR: %f' %
                  (epoch + 1, args.epochs, state['lr']))
            val_loss, val_kappa, val_auc, val_f1 = [0, 0, 0, 0]
            train_loss, train_acc = train(
                trainloader, model, criterion, optimizer, epoch, use_cuda)
            args.eval = True
            val_loss, val_acc = test(valloader, model, criterion, use_cuda)
            _, TTA_acc = single_test_TTA_hard(model, valloader, use_cuda, num_TTA=args.num_TTA, epoch=epoch + 1,
                                              savepath=args.checkpoint + dirname)

            # append logger file
            logger.append([state['lr'], train_loss.item(),
                          val_loss.item(), train_acc, val_acc, TTA_acc])

            # save model
            is_best = TTA_acc > best_acc
            if is_best:
                _, TTA_acc2 = single_test_TTA_hard(model, valloader, use_cuda, num_TTA=8, epoch=epoch + 1,
                                                   savepath=args.checkpoint + dirname)
                if TTA_acc2 > best_acc:
                    best_acc = min(TTA_acc, TTA_acc2)
                    logger_all.append([fold, best_acc, epoch])
                    torch.save(model.state_dict(),
                               args.checkpoint + dirname+f"/{fold}.pth")        
        model.load_state_dict(torch.load(args.checkpoint + dirname+f"/{fold}.pth"))
        testset = get_predictation_dataset_jpg_ti_label(size=args.input_size)
        testloader = data.DataLoader(
            testset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)
        res = single_test_TTA_hard_pred(model, testloader, use_cuda, num_TTA=8)
        predict(res,path=args.checkpoint + dirname+f"/{fold}_predict.csv")
        torch.cuda.empty_cache()
        logger.close()
        plt.clf()
        logger.plot()
        savefig(args.checkpoint + dirname + f'/{fold}.jpg')
    logger_all.close()


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
    miner= miners.MultiSimilarityMiner(epsilon=0.1)
    loss_func = TripletMarginLoss(margin=0.1)
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

    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    loss_func = TripletMarginLoss(margin=0.1)

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


if __name__ == '__main__':
    main()
