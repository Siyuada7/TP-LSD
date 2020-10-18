#!/usr/bin/env python3

import os
import sys
import glob
import os.path as osp

import numpy as np
import scipy.io
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2


image_p = "data/York/images/"
GT = "data/York/GT/"
suffix = '.png'

def msTPFP(line_pred, line_gt, threshold):
    diff = ((line_pred[:, None, :, None] - line_gt[:, None]) ** 2).sum(-1)
    diff = np.minimum(
        diff[:, :, 0, 0] + diff[:, :, 1, 1], diff[:, :, 0, 1] + diff[:, :, 1, 0]
    )
    choice = np.argmin(diff, 1)
    dist = np.min(diff, 1)
    hit = np.zeros(len(line_gt), np.bool)
    tp = np.zeros(len(line_pred), np.float)
    fp = np.zeros(len(line_pred), np.float)
    for i in range(len(line_pred)):
        if dist[i] < threshold and not hit[choice[i]]:
            hit[choice[i]] = True
            tp[i] = 1
        else:
            fp[i] = 1
    #print(tp.sum(), fp.sum())
    return tp, fp

def ap(tp, fp, name, t):
    for num in range(len(tp)):
        recall = tp[num]
        precision = tp[num] / np.maximum(tp[num] + fp[num], 1e-9)

        recall = np.concatenate(([0.0], recall, [1.0]))
        precision = np.concatenate(([0.0], precision, [0.0]))

        for i in range(precision.size - 1, 0, -1):
            precision[i - 1] = max(precision[i - 1], precision[i])
        i = np.where(recall[1:] != recall[:-1])[0]

        T = 0.005
        if len(recall) == 1:
            plt.scatter(recall[0], precision[0],  label=name[num])
        else:
            # f = interpolate.interp1d(recall, precision, kind="cubic", bounds_error=False)
            # x = np.arange(0, 1, 0.01) * recall[-1]
            # y = f(x)
            plt.plot(recall[recall>T], precision[recall>T], '-', linewidth=3, label=name[num])
        print(name[num], np.sum((recall[i + 1] - recall[i]) * precision[i + 1]))
    f_scores = np.linspace(0.2, 0.8, num=8)
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.3)
        plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4)
    plt.grid(True)
    plt.axis([0.0, 1.0, 0.0, 1.0])
    plt.xticks(np.arange(0, 1.0, step=0.1))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.yticks(np.arange(0, 1.0, step=0.1))
    plt.legend(loc=3)
    plt.title("PR Curve for APH " + str(t))
    plt.show()

    return np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

def line_score_mat(path, threshold=10, scale=320):
    preds = sorted(glob.glob(path))
    gts = sorted(glob.glob(GT))
    sumtp, sumfp = np.array([0]), np.array([0])
    n_gt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    for pred_name in preds:
        img = cv2.imread(image_p + pred_name.split('/')[-1].replace(".mat", suffix)) # wire jpg
        height, width = img.shape[:2]
        mat = sio.loadmat(pred_name)
        line_pred = mat["lines"]
        lcnn_score = mat["score"].reshape(-1)

        #with np.load(gt_name) as fgt:
            #gt_line = fgt["lpos"][:, :, :2]
        gt_name = GT + pred_name.split('/')[-1].replace(".mat", "_line.mat")
        mat = sio.loadmat(gt_name)
        gt_line = mat['lines'].reshape(-1, 2, 2)

        tmp = gt_line.copy()
        gt_line[:, 0, 0] = tmp[:, 0, 1] / height * 128
        gt_line[:, 0, 1] = tmp[:, 0, 0]/ width * 128
        gt_line[:, 1, 0] = tmp[:, 1, 1]/ height * 128
        gt_line[:, 1, 1] = tmp[:, 1, 0]/ width* 128
        n_gt += len(gt_line)

        line_pred = line_pred.reshape(-1, 2, 2)
        line_pred[:, :, 0] *= 128 / scale
        line_pred[:, :, 1] *= 128 / scale

        tmp = line_pred.copy()
        line_pred[:, 0, 0] = tmp[:, 0, 1] # y
        line_pred[:, 0, 1] = tmp[:, 0, 0] # x
        line_pred[:, 1, 0] = tmp[:, 1, 1]
        line_pred[:, 1, 1] = tmp[:, 1, 0]

        #
        #
        # img = cv2.resize(img, (128, 128))
        # for i in range(len(gt_line)):
        #     cv2.line(img, (int(gt_line[i, 0, 1]), int(gt_line[i, 0, 0])),
        #              (int(gt_line[i, 1, 1]), int(gt_line[i, 1, 0])), (255, 0, 0), 1)
        # plt.imshow(img)
        # plt.show()

        tp, fp = msTPFP(line_pred, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)
        sumtp[0] += tp.sum()
        sumfp[0] += fp.sum()

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)
    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt
    tps = sumtp
    fps = sumfp
    print(tps, fps)
    N = n_gt

    return lcnn_tp, lcnn_fp

def lcnn_score(path, lcnn=True, threshold=10):
    preds = sorted(glob.glob(path))
    gts = sorted(glob.glob(GT))

    sumtp, sumfp = np.array([0]), np.array([0])
    n_gt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    for pred_name in preds:
        img = cv2.imread(image_p + pred_name.split('/')[-1].replace(".npz", suffix))
        height, width = img.shape[:2]
        with np.load(pred_name) as fpred:
            lcnn_line = fpred["lines"][:, :, :2]
            lcnn_score = fpred["score"]

        for i in range(len(lcnn_line)):
            if i > 0 and (lcnn_line[i] == lcnn_line[0]).all():
                lcnn_line = lcnn_line[:i]
                lcnn_score = lcnn_score[:i]
                break
        if not lcnn:
            lcnn_line/=4

        gt_name = GT + pred_name.split('/')[-1].replace(".npz", "_line.mat")
        mat = sio.loadmat(gt_name)
        gt_line = mat['lines'].reshape(-1, 2, 2)

        tmp = gt_line.copy()
        gt_line[:, 0, 0] = tmp[:, 0, 1] / height * 128
        gt_line[:, 0, 1] = tmp[:, 0, 0] / width * 128
        gt_line[:, 1, 0] = tmp[:, 1, 1] / height * 128
        gt_line[:, 1, 1] = tmp[:, 1, 0] / width * 128
        n_gt += len(gt_line)

        # img = cv2.resize(img, (128, 128))
        # for i in range(len(gt_line)):
        #     cv2.line(img, (int(lcnn_line[i, 0, 1]), int(lcnn_line[i, 0, 0])),
        #              (int(lcnn_line[i, 1, 1]), int(lcnn_line[i, 1, 0])), (255, 0, 0), 1)
        # plt.imshow(img)
        # plt.show()

        tp, fp = msTPFP(lcnn_line, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)
        sumtp[0] += tp.sum()
        sumfp[0] += fp.sum()

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)
    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt
    # tps = sumtp
    # fps = sumfp
    # N = n_gt
    # rcs = sorted(list((tps / N)[:]))
    # prs = sorted(list((tps / np.maximum(tps + fps, 1e-9))[:]))[::-1]
    # print(ap(lcnn_tp, lcnn_fp))
    # return ap(lcnn_tp, lcnn_fp)
    return lcnn_tp, lcnn_fp

def afm_mat(path, threshold=10, reverse=False):
    preds = sorted(glob.glob(path))
    gts = sorted(glob.glob(GT))
    sumtp, sumfp = np.array([0]), np.array([0])
    n_gt = 0
    lcnn_tp, lcnn_fp, lcnn_scores = [], [], []
    for pred_name in preds:
        #print(pred_name)
        img = cv2.imread(image_p + pred_name.split('/')[-1].replace(".mat", suffix)) # wire jpg
        height, width = img.shape[:2]
        mat = sio.loadmat(pred_name)
        line_pred = mat["lines"]
        if reverse:
            lcnn_score = -mat["scores"].reshape(-1)
        else:
            lcnn_score = mat["scores"].reshape(-1)


        gt_name = GT + pred_name.split('/')[-1].replace(".mat", "_line.mat")
        mat = sio.loadmat(gt_name)
        gt_line = mat['lines'].reshape(-1, 2, 2)

        tmp = gt_line.copy()
        gt_line[:, 0, 0] = tmp[:, 0, 1] / height * 128
        gt_line[:, 0, 1] = tmp[:, 0, 0]/ width * 128
        gt_line[:, 1, 0] = tmp[:, 1, 1]/ height * 128
        gt_line[:, 1, 1] = tmp[:, 1, 0]/ width* 128
        n_gt += len(gt_line)

        line_pred = line_pred.reshape(-1, 2, 2)
        line_pred[:, :, 0] *= 128 / width
        line_pred[:, :, 1] *= 128 / height

        tmp = line_pred.copy()
        line_pred[:, 0, 0] = tmp[:, 0, 1] # y
        line_pred[:, 0, 1] = tmp[:, 0, 0] # x
        line_pred[:, 1, 0] = tmp[:, 1, 1]
        line_pred[:, 1, 1] = tmp[:, 1, 0]

        # img = cv2.resize(img, (128, 128))
        # for i in range(len(line_pred)):
        #     cv2.line(img, (int(line_pred[i, 0, 1]), int(line_pred[i, 0, 0])),
        #              (int(line_pred[i, 1, 1]), int(line_pred[i, 1, 0])), (255, 0, 0), 1)
        # plt.imshow(img)
        # plt.show()
        # sys.exit()

        tp, fp = msTPFP(line_pred, gt_line, threshold)
        lcnn_tp.append(tp)
        lcnn_fp.append(fp)
        lcnn_scores.append(lcnn_score)
        sumtp[0] += tp.sum()
        sumfp[0] += fp.sum()

    lcnn_tp = np.concatenate(lcnn_tp)
    lcnn_fp = np.concatenate(lcnn_fp)
    lcnn_scores = np.concatenate(lcnn_scores)
    lcnn_index = np.argsort(-lcnn_scores)
    lcnn_tp = np.cumsum(lcnn_tp[lcnn_index]) / n_gt
    lcnn_fp = np.cumsum(lcnn_fp[lcnn_index]) / n_gt
    tps = sumtp
    fps = sumfp
    #print(tps, fps)
    N = n_gt

    #print(ap(lcnn_tp, lcnn_fp))
    #return ap(lcnn_tp, lcnn_fp)
    return lcnn_tp, lcnn_fp

if __name__ == "__main__":
    # afm = '/AFM/york/atrous/scores/*.mat'
    # lsd = '/LSD/york/lsd_scores/*.mat'
    # wire = '/DWP/york/'
    # lcnn = '/LCNN/york/logs/pretrained-model/npz/000312000/'
    # lcnn_p = '/LCNN/york/post/pretrained-model-000312000/0_010/'

    test = 'log/test/york/mat/lmbd0.5/0.01/*.mat'

    path = [test]
    name = ['test']

    tp, fp = [0 for i in range(len(name))], [0 for i in range(len(name))]
    for t in range(10, 11):
        print('-----t----', t)
        for i in range(len(path)):
            print(path[i])
            tp[i], fp[i] = line_score_mat(path[i], t, scale=320)
            # tp[i], fp[i] = afm_mat(afm, t, reverse=True)
            # tp[i], fp[i] = afm_mat(lsd, t, reverse=False)
            # tp[i], fp[i] = wireframe_score(wire, t)
            # tp[i], fp[i] = lcnn_score(lcnn, True, t)
        ap(tp, fp, name, t)
    sys.exit()
