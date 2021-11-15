from __future__ import print_function, absolute_import
from .quadratic_weighted_kappa import quadratic_weighted_kappa
from sklearn import metrics
import numpy as np

__all__ = ['accuracy', 'conf_matrix', 'qw_kappa', 'odir_metrics']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def conf_matrix(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    rater_a = pred.squeeze().detach().numpy()
    rater_b = target.view(1, -1).squeeze().detach().numpy()
    conf_mat = metrics.confusion_matrix(rater_b, rater_a, labels=[0, 1, 2, 3, 4])
    return conf_mat


def qw_kappa(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred.squeeze().tolist())
    # print(target.view(1, -1).squeeze().tolist())
    kappa = quadratic_weighted_kappa(pred.squeeze().tolist(), target.view(1, -1).squeeze().tolist())
    return kappa


def odir_metrics(pr_data, gt_data):
    th = 0.5
    gt = gt_data.flatten()
    pr = pr_data.flatten()
    kappa = metrics.cohen_kappa_score(gt, pr > th)
    f1 = metrics.f1_score(gt, pr > th, average='micro')
    auc = metrics.roc_auc_score(gt, pr)
    final_score = (kappa + f1 + auc) / 3.0

    batch_size = gt_data.size(0)
    acc_each_label = np.zeros(5)
    for i in range(batch_size):
        # print(gt_data.numpy())
        row_gt = np.array(gt_data)[i, :]
        row_pr = np.array(pr_data)[i, :]
        row_pr[row_pr >= 0.5] = 1
        row_pr[row_pr < 0.5] = 0
        row = np.zeros(5)
        row[row_gt == row_pr] = 1
        acc_each_label += row
    acc_each_label = acc_each_label / batch_size
    # print(acc_each_label)

    return kappa, f1, auc, final_score, acc_each_label


def ti_metrics_single_class(pr_data, gt_data):
    batch_size = gt_data.size(0)
    acc_each_label = np.zeros(5)
    correct_count = 0

    for i in range(batch_size):
        pr_label = pr_data[i].argmax()
        gt_label = gt_data[i].argmax()

        if pr_label == gt_label:
            correct_count = correct_count + 1

    acc = float(correct_count) / float(batch_size)

    return acc

# def ti_metrics_single_class_label(pr_data, gt_data):
#     batch_size = gt_data.size(0)
#     correct_count = 0
#     for i in range(batch_size):
#         pr_label = pr_data[i].argmax()
#         gt_label = gt_data[i]
#         print(pr_data[i])
#         print(pr_label)
#         print(gt_label)
#         if pr_label == gt_label:
#             correct_count = correct_count + 1
    
#     acc = float(correct_count) / float(batch_size)

#     return acc