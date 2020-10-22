import torch.nn as nn
import os
import time
import torch
import torchvision
import numpy as np
import cv2 as cv
import scipy.io as sio

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)

    keep = (hmax == heat).float()
    heat =  heat * keep
    return heat

def TPS_line(output, thresh=0.2, lmbd=0, H=320, W=320):
    pred_displacement_test = output['dis'][0]
    line_test = output['line']
    center_test = output['center'] * (line_test ** lmbd)
    center_nms = _nms(center_test, kernel=5)
    pos_test = center_nms[0][0].gt(thresh).nonzero()
    dis_list_test = pred_displacement_test[:, pos_test[:, 0], pos_test[:, 1]].transpose(1, 0)
    pos_test[:, [0, 1]] = pos_test[:, [1, 0]]
    pos_test = pos_test.float()
    start_point_test = pos_test + dis_list_test[:, 0:2]
    end_point_test = pos_test + dis_list_test[:, 2:4]

    start_point = start_point_test.detach().cpu().numpy()
    end_point = end_point_test.detach().cpu().numpy()

    line = np.stack([start_point, end_point], axis=1)
    torch.cuda.synchronize()
    endtime = time.time()
    return line.reshape((-1, 4)), start_point, end_point, pos_test.cpu().numpy(), endtime

def save_pic_mat(lmbd, thresh, outputs, image, filename, log_path, save_mat=False, save_pic=True):
    output = outputs[-1]
    H, W = output['center'].shape[2:4]
    lines, start_point, end_point, pos, endtime = TPS_line(output, thresh, lmbd, H, W)

    if save_mat:
        center = output['center'][0][0].detach().cpu().numpy()
        pos_mat = pos.astype(np.int)
        if len(pos) > 0:
            savescorelist = center.copy()
            savescorelist = savescorelist[pos_mat[:, 1], pos_mat[:, 0]].tolist()
        else:
            savescorelist = []
        savelinelist = lines
        mat_dir = log_path + '/mat/' + '/lmbd' + str(lmbd) + '/' + str(thresh)
        os.makedirs(mat_dir, exist_ok=True)
        mat_name = mat_dir + '/' + filename + '.mat'
        sio.savemat(mat_name, {"lines": savelinelist, 'score': savescorelist})
    if save_pic:
        center_list = pos
        img_pred = image[0].cpu().numpy()
        img_pred = cv.resize(img_pred, (H, W))
        center = output['center'][0][0].detach().cpu().numpy()
        line_hm = output['line'][0][0].detach().cpu().numpy()
        for i in range(center_list.shape[0]):
            center_coor = (center_list[i][0], center_list[i][1])
            start_coor = (int(start_point[i][0]), int(start_point[i][1]))
            end_coor = (int(end_point[i][0]), int(end_point[i][1]))

            cv.line(img_pred, start_coor, end_coor, (255, 255, 0), 1)
            cv.circle(img_pred, start_coor, 2, (255, 0, 0), -1)
            cv.circle(img_pred, end_coor, 2, (0, 255, 0), -1)  # green
            cv.circle(img_pred, center_coor, 2, (0, 0, 255), -1)  # red
        center_cat = np.expand_dims(center * 255., 2).repeat(3, axis=2)
        line_cat = np.expand_dims(line_hm * 255., 2).repeat(3, axis=2)
        return img_pred, line_cat, center_cat, endtime
    else:
        return endtime

def save_image(lmbd, thresh, outputs, image):
    output = outputs[-1]
    H, W = output['center'].shape[2:4]
    lines, start_point, end_point, pos, endtime = TPS_line(output, thresh, lmbd, H, W)
    img_pred = image[0].numpy()
    W_ = img_pred.shape[1] / W
    H_ = img_pred.shape[0] / H
    lines[:, [0, 2]] *= W_
    lines[:, [1, 3]] *= H_

    for i in range(lines.shape[0]):
        start_coor = (int(round(lines[i][0])), int(round(lines[i][1])))
        end_coor = (int(round(lines[i][2])), int(round(lines[i][3])))
        cv.line(img_pred, start_coor, end_coor, (110, 215, 245), 2, lineType=16)
        cv.circle(img_pred, start_coor, 3, (234, 245, 134), -1, lineType=16)
        cv.circle(img_pred, end_coor, 3, (234, 245, 134), -1, lineType=16)

    return img_pred, lines, endtime