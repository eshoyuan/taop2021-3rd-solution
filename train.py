from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import timm
from timm.data.transforms_factory import create_transform
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets.get_dataset import get_dataset_jpg_ti, get_dataset_jpg_ti_onehot, get_predictation_dataset_jpg_ti_label
from utils import Bar, Logger, AverageMeter, odir_metrics, mkdir_p, savefig
from utils.eval import ti_metrics_single_class
from torch.utils.checkpoint import checkpoint_sequential
import matplotlib.pyplot as plt
import csv
from pytorch_metric_learning import miners, reducers, testers
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator, precision_at_k

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def set_parameter_requires_grad(model, feature_finetuning):
    if feature_finetuning:
        for param in model.parameters():
            param.requires_grad = True


class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.05, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def set_parameter_requires_grad(model, feature_finetuning):
    if feature_finetuning:
        for param in model.parameters():
            param.requires_grad = True


parser = argparse.ArgumentParser(
    description='PyTorch Single-label Classification Training - Ocular Single-Disease Identification')
parser.add_argument(
    '--dataset', default='/mnt/data1/MedicalDataset/taop-2021/png_crop', type=str, help='dataset path')
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
parser.add_argument('--schedule', type=int, nargs='+', default=[8, 17, 25, 30, 33],
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
parser.add_argument('-c', '--checkpoint', default='./kdcheckpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--pretrained', default=True, type=bool,
                    help='whether to load model pretrained on Imagenet')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='repvgg_b3g4',
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')

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
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    start_time = int(time.time())
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data
    print('==> Preparing dataset')
    # Choose backbone model
    print("==> creating model '{}'".format(args.arch))
    model = timm.create_model(
        args.arch, pretrained=args.pretrained, num_classes=5)

    # 使得模型能运行在两块显卡上，若使用单卡则需要注释下面这一行
    model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters()) / 1000000.0))

    # 选择backbone model的loss function
    criterion = decide_model_criterion()
    criterion = LabelSmoothSoftmaxCEV1(lb_smooth=0.1)
    if args.optim == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr, momentum=args.momentum)

    if args.format_label == 'scalar':
        trainset, valset, val_image_list = get_dataset_jpg_ti(
            dataset_path=args.dataset, input_size=args.input_size)
    if args.format_label == 'onehot':
        trainset, valset, val_image_list = get_dataset_jpg_ti_onehot(dataset_path=args.dataset,
                                                                     input_size=args.input_size)

    trainloader = data.DataLoader(
        trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
    valloader = data.DataLoader(
        valset, batch_size=args.train_batch, shuffle=False, num_workers=args.workers)

    # Resume
    title = 'DR-GRADING-' + args.arch
    # 创建存储混淆矩阵图片的文件夹
    dirname = f'/{args.arch}_lr{args.lr}_bz{args.train_batch}_size{args.input_size}_pt{args.pretrained}_{start_time}'
    os.makedirs(args.checkpoint + str(dirname), mode=0o777)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(
            args.checkpoint + dirname +
                f'/log_{args.arch}_lr{args.lr}_bz{args.train_batch}_size{args.input_size}_{start_time}.txt',
            title=title,
            resume=True)
    else:
        logger = Logger(
            args.checkpoint + dirname +
                f'/log_{args.arch}_lr{args.lr}_bz{args.train_batch}_size{args.input_size}_{start_time}.txt',
            title=title,
            resume=False)
        logger.set_names(
            ['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc', 'Val Acc', 'TTA Acc'])

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' %
              (epoch + 1, args.epochs, state['lr']))

        val_loss, val_kappa, val_auc, val_f1 = [0, 0, 0, 0]
        # optimizer_warmup = optim.Adam(model.module.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # if epoch<3:
        #     train_loss, train_acc = train(trainloader, model, criterion, optimizer_warmup, epoch, use_cuda)
        # else:
        train_loss, train_acc = train(
            trainloader, model, criterion, optimizer, epoch, use_cuda)
        args.eval = True
        val_loss, val_acc = test(valloader, model, criterion, use_cuda)
        _, TTA_acc = single_test_TTA_soft(model, valloader, use_cuda, num_TTA=args.num_TTA, epoch=epoch + 1,
                                                   savepath=args.checkpoint + dirname)

        # append logger file
        logger.append([state['lr'], train_loss.item(),
                      val_loss.item(), train_acc, val_acc, TTA_acc])
        if TTA_acc > 0.85:
            _, TTA_acc = single_test_TTA_soft(model, valloader, use_cuda, num_TTA=args.num_TTA, epoch=epoch + 1,
                                                   savepath=args.checkpoint + dirname)
            if TTA_acc > 0.85:
                torch.save(model.state_dict(),"seresnext50.pth",_use_new_zipfile_serialization=False)
                testset = get_predictation_dataset_jpg_ti_label(size=args.input_size)
                testloader = data.DataLoader(testset, batch_size = args.train_batch, shuffle = False, num_workers = args.workers)
                res=single_test_TTA_soft_pred(model, testloader, use_cuda)
                predict(res)
                return 
                  # save model
        # is_best = val_kappa > best_acc
        # best_acc = max(val_kappa, best_acc)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model.state_dict(),
        #     'train_acc': train_acc,
        #     'val_acc': val_acc,
        #     'optimizer': optimizer.state_dict(),
        # }, 1, checkpoint=args.checkpoint)

    logger.close()
    plt.clf()
    logger.plot()
    savefig(args.checkpoint + dirname + '/pic_{}_lr{}_bz{}_size{}_{}.jpg'.format(args.arch, args.lr, args.train_batch,
                                                                                 args.input_size, int(time.time())))



def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time=AverageMeter()
    data_time=AverageMeter()
    losses=AverageMeter()
    losses_m=AverageMeter()
    acc=np.zeros(args.num_labels)
    score_acc=AverageMeter()
    end=time.time()
    # metrci learning
    miner=miners.MultiSimilarityMiner()
    loss_func=TripletMarginLoss()
    bar=Bar('Processing', max = len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)



        targets_list=targets.tolist()

        if use_cuda:
            inputs, targets=inputs.cuda(), targets.cuda()




        # compute output
        if args.arch == 'inception_v3':
            outputs, aux=model(inputs)
            loss1=criterion(outputs, targets)
            loss2=criterion(aux, targets)
            loss=loss1 + 0*loss2
        else:
            outputs=model(inputs)
            hard_pairs=miner(outputs, targets)
            loss2=loss_func(outputs, targets, hard_pairs)

            loss1=criterion(outputs, targets)
            loss=loss1 + 0.1 * loss2

        acc=compute_acc(outputs, targets_list)
        score_acc.update(acc, inputs.size(0))
        losses_m.update(loss2.data, inputs.size(0))
        losses.update(loss1.data, inputs.size(0))

        # compute gradient and do AdamW step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end=time.time()

        bar.suffix='({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_m: {loss2:.4f} | ACC: {score_acc: .4f}'.format(
            batch = batch_idx + 1,
            size = len(trainloader),
            data = data_time.avg,
            bt = batch_time.avg,
            total = bar.elapsed_td,
            eta = bar.eta_td,
            loss = losses.avg,
            loss2 = losses_m.avg,
            score_acc = score_acc.avg,
        )
        bar.next()
    bar.finish()

    # acc = acc / n_train
    # print(acc)
    return (losses.avg, score_acc.avg)


def test(testloader, model, criterion, use_cuda):
    global best_acc


    miner=miners.MultiSimilarityMiner()
    loss_func=TripletMarginLoss()

    batch_time=AverageMeter()
    data_time=AverageMeter()
    losses=AverageMeter()
    acc=np.zeros(args.num_labels)
    score_acc=AverageMeter()
    losses_m=AverageMeter()

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

    end=time.time()
    bar=Bar('Processing', max = len(testloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            # measure data loading time
            data_time.update(time.time() - end)
            targets_list=targets.tolist()
            if use_cuda:
                inputs, targets=inputs.cuda(), targets.cuda()

            # compute output
            if args.arch == 'inception_v3':
                outputs, aux=model(inputs)
                loss1=criterion(outputs, targets)
                loss2=criterion(aux, targets)
                loss=loss1 + loss2
            else:
                outputs=model(inputs)
                hard_pairs=miner(outputs, targets)
                loss2=loss_func(outputs, targets, hard_pairs)

                loss1=criterion(outputs, targets)
                # loss = loss1+ 0.3 *loss2

            acc=compute_acc(outputs, targets_list)
            score_acc.update(acc, inputs.size(0))
            losses_m.update(loss2.data, inputs.size(0))
            losses.update(loss1.data, inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end=time.time()

            # plot progress
            bar.suffix='({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_m: {loss2:.4f} | ACC: {score_acc: .4f}'.format(
                            batch = batch_idx + 1,
                            size = len(testloader),
                            data = data_time.avg,
                            bt = batch_time.avg,
                            total = bar.elapsed_td,
                            eta = bar.eta_td,
                            loss = losses.avg,
                            loss2 = losses_m.avg,
                            score_acc = score_acc.avg,
                            )
            bar.next()
        bar.finish()
    return (losses.avg, score_acc.avg)


def save_checkpoint(state, is_best, checkpoint = 'checkpoint', filename = 'checkpoint.pth.tar'):
    filepath=os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(
            checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    # if epoch<3:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = 1e-5
    # if epoch==3:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = args.lr
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr']=state['lr']




def single_test_TTA_hard(model, testloader, use_cuda, epoch, savepath, num_TTA = 8):
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

    res=[]
    targets_list=[]
    all_targets=torch.tensor(2.5)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            all_targets=targets.clone()
        else:
            all_targets=torch.cat((all_targets, targets), 0)
    # print(all_targets.shape)

    for i in range(num_TTA):
        res.append([])
    with torch.no_grad():
        for n in range(num_TTA):  # TTA的次数
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets=inputs.cuda(), targets.cuda()
                if n == 0:
                    if args.format_label == 'scalar':
                        targets_list += targets.tolist()
                    if args.format_label == 'onehot':
                        targets_list += targets.argmax(dim = 1).tolist()
                outs=model(inputs)
                for i in range(outs.shape[0]):
                    res[n].append(outs.argmax(dim=1)[i].item())
    vote_res=[]

    for i in range(len(res[0])):
        temp=[]
        for j in range(num_TTA):
            temp.append(res[j][i])
        vote_res.append(np.argmax(np.bincount(temp)).item())
    # cm = confusion_matrix(targets_list, vote_res, labels=[0, 1, 2, 3, 4])
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    # disp.plot()
    # plt.savefig(savepath + f'/confusion_matrix_{epoch}.png')
    # predicts_onehot = convert_label_to_onehot(vote_res)
    # print(predicts_onehot.shape)
    acc=(np.array(vote_res) == np.array(targets_list)).sum() / len(targets_list)

    print("TTA_acc = ", acc)
    return use_cuda, acc

def single_test_TTA_soft(model, testloader, use_cuda, epoch, savepath, num_TTA = 8):
    """TTA+软投票, num_TTA为进行TTA的次数"""
    global best_acc

    if args.eval == False:
        model.train()
    else:
        model.eval()


    res=[]
    targets_list=[]
    all_targets=torch.tensor(2.5)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            all_targets=targets.clone()
        else:
            all_targets=torch.cat((all_targets, targets), 0)

    for i in range(num_TTA):
        res.append([])

    with torch.no_grad():
        totaloutputs=None
        for n in range(num_TTA):  # TTA的次数
            alloutputs=None
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets=inputs.cuda(), targets.cuda()
                if n == 0:
                    if args.format_label == 'scalar':
                        targets_list += targets.tolist()
                    if args.format_label == 'onehot':
                        targets_list += targets.argmax(dim = 1).tolist()
                outs=model(inputs)
                if alloutputs == None:
                    alloutputs=outs
                else:
                    alloutputs=torch.cat((alloutputs, outs), 0)
            if totaloutputs == None:
                totaloutputs=alloutputs
            else:
                totaloutputs += alloutputs
    totaloutputs /= num_TTA
    acc=compute_acc(totaloutputs, targets_list)
    print("TTA_acc = ", acc)
    return use_cuda, acc

# 决定backbone model使用的loss function
def decide_model_criterion():
    criterion=None
    if args.criterion.lower() == 'bcewithlogitsloss':
        # 若使用BCEWithLogitsLoss作为loss function，那么labels应为onehot的形式
        args.format_label='onehot'

        #     use BCEWithLogitsLoss as criterion with one-hot
        if args.weighted_CE == 1:
            criterion=nn.BCEWithLogitsLoss(weight = class_weights.cuda())
        else:
            criterion=nn.BCEWithLogitsLoss()
            print('weights = [1 1 1 1 1]')

    elif args.criterion.lower() == 'crossentropyloss':
        # 若使用CrossEntropyLoss作为loss function，那么labels应为scalar的形式
        args.format_label='scalar'

        # use CrossEntropyLoss as criterion with non one-hot labels
        if args.weighted_CE == 1:
            criterion=nn.CrossEntropyLoss(weight = class_weights.cuda())
        else:
            criterion=nn.CrossEntropyLoss()
            print('weights = [1 1 1 1 1]')

    return criterion

# 由给定activate function计算acc
def compute_acc(outputs, targets_list):
    # 其实这里不需要区分激活函数, softmax和sigmoid都是单调函数, 取outputs最大即可
    acc=None
    if args.activate.lower() == 'softmax':
        if args.format_label == 'scalar':
            pred=torch.softmax(outputs, 1).detach().cpu().numpy()
            res=[]
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            acc=(np.array(res) == np.array(targets_list)
                 ).sum() / len(targets_list)
        if args.format_label == 'onehot':
            pred=torch.softmax(outputs, 1).detach().cpu().numpy()
            res=[]
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            # 若target的形式为onehot，则转换为scalar
            targets_list_scalar=[]
            for i in range(len(targets_list)):
                targets_list_scalar.append(np.argmax(targets_list[i]))
            acc=(np.array(res) == np.array(targets_list_scalar)
                 ).sum() / len(targets_list)

    elif args.activate.lower() == 'sigmoid':
        if args.format_label == 'scalar':
            pred=torch.sigmoid(outputs).detach().cpu().numpy()
            res=[]
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            acc=(np.array(res) == np.array(targets_list)
                 ).sum() / len(targets_list)
        if args.format_label == 'onehot':
            pred=torch.sigmoid(outputs).detach().cpu().numpy()
            res=[]
            for i in range(len(pred)):
                res.append(np.argmax(pred[i]))
            # 若target的形式为onehot，则转换为scalar
            targets_list_scalar=[]
            for i in range(len(targets_list)):
                targets_list_scalar.append(np.argmax(targets_list[i]))
            acc=(np.array(res) == np.array(targets_list_scalar)
                 ).sum() / len(targets_list)
    return acc

def predict(res):
    # switch to train mode
    file = '/mnt/data1/MedicalDataset/taop-2021/test2_data_info.csv'
    data = pd.read_csv(file, encoding='utf-8')
    for i in range(len(data)):
        data[u'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'][i] = str(int(res[i]+1))
        data[u'id_project']=str(100001)
    data[u'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'] = data[u'class(1:PM;2:MD;3:glaucoma;4:RVO;5:DR)'].astype(int)
    data[u'id_project']=data[u'id_project'].astype(int)
    pathname = str(int(time.time()))
    data.to_csv("./results_seresnext50.csv", index=False, encoding='utf-8')
    
def single_test_TTA_soft_pred(model, testloader, use_cuda, num_TTA = 8):
    """TTA+软投票, num_TTA为进行TTA的次数"""
    global best_acc

    if args.eval == False:
        model.train()
    else:
        model.eval()

    res=[]
    targets_list=[]
    all_targets=torch.tensor(2.5)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            all_targets=targets.clone()
        else:
            all_targets=torch.cat((all_targets, targets), 0)

    for i in range(num_TTA):
        res.append([])

    with torch.no_grad():
        totaloutputs=None
        for n in range(num_TTA):  # TTA的次数
            alloutputs=None
            for batch_idx, (inputs, targets) in enumerate(testloader):
                if use_cuda:
                    inputs, targets=inputs.cuda(), targets.cuda()
                if n == 0:
                    targets_list += targets.tolist()
                outs=model(inputs)
                if alloutputs == None:
                    alloutputs=outs
                else:
                    alloutputs=torch.cat((alloutputs, outs), 0)
            if totaloutputs == None:
                totaloutputs=alloutputs
            else:
                totaloutputs += alloutputs
    totaloutputs /= num_TTA
    res=[]
    for i in range(totaloutputs.shape[0]):
        res.append(totaloutputs.argmax(dim=1)[i].item())
    return res


if __name__ == '__main__':
    main()
