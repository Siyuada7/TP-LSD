import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import os
import cv2 as cv
import torch.utils.data as data
import random
import copy
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap


class Wireframe(data.Dataset):
    def __init__(self, split, param):
        super(Wireframe, self).__init__()
        self.param = param
        self.data_dir = os.path.join(param.dataset_dir, 'coco')
        if split == 'test':
            self.img_dir = self.data_dir
            self.images = os.listdir(self.img_dir)
        else:
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_{}2017.json'.format(split))
            self.img_dir = os.path.join(self.data_dir, 'images', '{}2017'.format(split))
            self.coco = coco.COCO(self.annot_path) 
            self.images = self.coco.getImgIds()

        self.num_samples = len(self.images)
        self.split = split
        self.in_res = self.param.inres
        self.out_res = self.param.outres
        self.down_scale1 = 512 / self.in_res[0]
        self.down_scale = self.in_res[0] / self.out_res[0]
        self.length_thresh = self.param.length_thres
        self.centerweight = self.param.center_weight
        self.center_other_weight = self.param.center_other_weight
        self.lineweight = self.param.line_weight
        self.linewidth = self.param.line_width

        self.g_kernel = np.array([[0.02777778, 0.11111111, 0.16666667, 0.11111111, 0.02777778],
                             [0.11111111, 0.44444444, 0.66666667, 0.44444444, 0.11111111],
                             [0.16666667, 0.66666667, 1., 0.66666667, 0.16666667],
                             [0.11111111, 0.44444444, 0.66666667, 0.44444444, 0.11111111],
                             [0.02777778, 0.11111111, 0.16666667, 0.11111111, 0.02777778]])

        print('==> initializing coco 2017 {} data.'.format(split))
        print('Loaded {} {} samples'.format(self.num_samples, split))

    def __len__(self):
        return self.num_samples

    def point_rot_pro(self, bbox):
        assert isinstance(bbox, np.ndarray)
        hor = 0
        ver = 0
        # Find the intersection point with boundary
        if bbox[3] == bbox[1]:
            x1, x2, y1, y2 = -1, -1, 0, self.out_res[1] - 1
            hor = 1
        elif bbox[2] == bbox[0]:
            x1, x2, y1, y2 = 0, self.out_res[0] - 1, -1, -1
            ver = 1
        else:
            slop = (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])
            # left
            y1 = - bbox[2] * slop + bbox[3]
            # right
            y2 = (self.out_res[0] - 1 - bbox[2]) * slop + bbox[3]
            # up
            x1 = - bbox[3] / slop + bbox[2]
            # down
            x2 = (self.out_res[0] - 1 - bbox[3]) / slop + bbox[2]
        junc = np.array([y1, y2, x1, x2])
        junc_flag1 = np.where(junc >= 0, 1, 0)
        junc_flag2 = np.where(junc <= self.out_res[0] - 1, 1, 0)
        junc_flag = junc_flag1 * junc_flag2

        if junc_flag.sum() < 2:
            return False
        else:
            if hor == 1:
                if bbox[0] < 0:
                    bbox[0] = 0
                if bbox[2] >= self.out_res[0] - 1:
                    bbox[2] >= self.out_res[0] - 1
            elif ver == 1:
                if bbox[1] < 0:
                    bbox[1] = 0
                if bbox[3] >= self.out_res[0] - 1:
                    bbox[3] = self.out_res[0] - 1
            elif junc_flag[0] == 1 and junc_flag[3] == 1:
                if bbox[0] < 0:
                    bbox[0] = 0
                    bbox[1] = y1
                if bbox[3] >= self.out_res[0] - 1:
                    bbox[2] = x2
                    bbox[3] = self.out_res[0] - 1
            elif junc_flag[0] == 1 and junc_flag[2] == 1:
                if bbox[0] < 0:
                    bbox[0] = 0
                    if ver == 0:
                        bbox[1] = y1
                if bbox[3] < 0:
                    bbox[3] = 0
                    if ver == 0:
                        bbox[2] = x1
            elif junc_flag[0] == 1 and junc_flag[1] == 1:
                if bbox[0] < 0:
                    bbox[0] = 0
                    bbox[1] = y1
                if bbox[2] >= self.out_res[0] - 1:
                    bbox[2] = self.out_res[0] - 1
                    bbox[3] = y2
            elif junc_flag[2] == 1 and junc_flag[3] == 1:
                if bbox[1] > bbox[3]:
                    if bbox[1] >= self.out_res[0] - 1:
                        bbox[1] = self.out_res[0] - 1
                        bbox[0] = x2
                    if bbox[3] < 0:
                        bbox[2] = x1
                        bbox[3] = 0
                else:
                    if bbox[1] < 0:
                        bbox[1] = 0
                        bbox[0] = x1
                    if bbox[3] >= self.out_res[0] - 1:
                        bbox[2] = x2
                        bbox[3] = self.out_res[0] - 1

            elif junc_flag[2] == 1 and junc_flag[1] == 1:
                if bbox[1] < 0:
                    bbox[1] = 0
                    bbox[0] = x1
                if bbox[2] >= self.out_res[0] - 1:
                    bbox[3] = y2
                    bbox[2] = self.out_res[0] - 1
            elif junc_flag[3] == 1 and junc_flag[1] == 1:
                if bbox[1] >= self.out_res[0] - 1:
                    bbox[0] = x2
                    bbox[1] = self.out_res[0] - 1
                if bbox[2] >= self.out_res[0] - 1:
                    bbox[2] = self.out_res[0] - 1
                    bbox[3] = y2
        if np.sqrt(((bbox[[0, 1]] - bbox[[2, 3]]) ** 2).sum()) < self.length_thresh:
            return False
        else:
            return True

    def imgnormalize(self, img):
        H, W, C = img.shape

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        imgv0 = hsv[..., 2]
        imgv = cv.resize(imgv0, (0, 0), fx=1. / 4, fy=1. / 4, interpolation=cv.INTER_LINEAR)
        imgv = cv.GaussianBlur(imgv, (5, 5), 3)
        imgv = cv.resize(imgv, (W, H), interpolation=cv.INTER_LINEAR)
        imgv = cv.GaussianBlur(imgv, (5, 5), 3)

        imgv1 = imgv0.astype(np.float32) - imgv + 127.5
        imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
        hsv[..., 2] = imgv1
        img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        inp = np.array(img, dtype=np.float32) / 255.0

        return inp

    def decode(self, bbox):
        x1, y1, x2, y2 = bbox.tolist()
        centerx, centery = (x1 + x2) / 2, (y1 + y2) / 2
        return centerx, centery, bbox.tolist()

    def draw_center_point(self, int_x, int_y, center_hm, bbox_use, kp_displacement, line):
        size = self.out_res[0]

        local_mask = np.zeros_like(center_hm)
        # draw center point & local mask
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if 0 <= int_y + dy < size and 0 <= int_x + dx < size:
                    center_hm[0][int_y + dy, int_x + dx] = max(self.g_kernel[2 + dy, 2 + dx], center_hm[0][int_y + dy, int_x + dx])
                    local_mask[0][int_y + dy, int_x + dx] = 1

        pos = np.where(local_mask[0] > 1e-3)
        x = pos[1]
        y = pos[0]
        start_point = [bbox_use[0], bbox_use[1]]
        end_point = [bbox_use[2], bbox_use[3]]
        cv.line(line[0], (int(start_point[0]), int(start_point[1])), (int(end_point[0]), int(end_point[1])),
                1, self.linewidth, lineType=cv.LINE_AA)
        self.draw_displacement(x, y, start_point, 0, kp_displacement)
        self.draw_displacement(x, y, end_point, 1, kp_displacement)

    def draw_displacement(self, x, y, end_point, index, kp_displacement):  # index=0起点，index=1端点
        x_offset = end_point[0] - x
        y_offset = end_point[1] - y
        kp_displacement[index * 2, y, x] = x_offset
        kp_displacement[index * 2 + 1, y, x] = y_offset


    def data_aug(self, img, bboxs):
        bboxs = bboxs.tolist()
        is_affine = random.random()
        is_scale = random.random()
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            # if use the standard augmentation operation, comment Affine
            iaa.Affine(rotate=(-90, 90), mode='constant') if is_affine < 0.6 else iaa.Sequential(),
            iaa.Affine(scale=(0.6, 1.5), mode='constant') if is_scale < 0.8 else iaa.Sequential(),
        ])

        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_image(img)

        new_bboxs = []
        kps_ori = np.reshape(np.asarray(bboxs), newshape=(len(bboxs), 2, 2)) if bboxs is not None else None
        kps = ia.KeypointsOnImage([], shape=img.shape)
        assert isinstance(bboxs, list)
        for i in range(len(kps_ori)):
            kps.keypoints.append(ia.Keypoint(x=kps_ori[i][0][0], y=kps_ori[i][0][1]))
            kps.keypoints.append(ia.Keypoint(x=kps_ori[i][1][0], y=kps_ori[i][1][1]))
        kps_aug = seq_det.augment_keypoints(kps)
        for i in range(len(kps_aug.keypoints)):
            point = kps_aug.keypoints[i]
            new_bboxs.append([point.x, point.y])
        new_bboxs = np.array(new_bboxs).reshape((len(bboxs), 4))

        return img_aug, new_bboxs

    def __getitem__(self, item):
        img_id = self.images[item]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name + '.png')
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        inp = cv.resize(cv.imread(img_path), self.in_res)

        bbox_ = np.array([anns[i]['bbox'] for i in range(len(anns))])
        len_ = np.sqrt((bbox_[:, 0] - bbox_[:, 2]) ** 2 + (bbox_[:, 1] - bbox_[:, 3]) ** 2)
        sort = np.argsort(len_)
        bbox_, len_ = bbox_[sort], len_[sort]
        bbox_ = bbox_[np.where(len_ > self.length_thresh)]
        bbox_ /= self.down_scale1
        if self.split == 'train':
            inp, bbox_ = self.data_aug(inp, bbox_)
        bbox_ /= self.down_scale

        inp = self.imgnormalize(inp).transpose((2, 0, 1))

        center_hm = np.zeros((1, self.out_res[0], self.out_res[1]), dtype=np.float32)
        kp_displacement = np.zeros((4, self.out_res[0], self.out_res[1]), dtype=np.float32)
        kp_mask = np.zeros((1, self.out_res[0], self.out_res[1]), dtype=np.float32)
        center_mask = np.ones((1, self.out_res[0], self.out_res[1]), dtype=np.float32)
        line = np.zeros((1, self.out_res[0], self.out_res[1]), dtype=np.float32)
        line_mask = np.ones((1, self.out_res[0], self.out_res[1]), dtype=np.float32)
        for k in range(len(bbox_)):
            bbox = bbox_[k].copy()
            if ((bbox[0] - bbox[2]) ** 2 + (bbox[1] - bbox[3]) ** 2) < self.length_thresh ** 2:
                continue
            if (bbox[0] < 0 and bbox[2] < 0) or (bbox[0] >= self.out_res[0] and bbox[2] >= self.out_res[0]):
                continue
            elif (bbox[1] < 0 and bbox[3] < 0) or (bbox[1] >= self.out_res[0] and bbox[3] >= self.out_res[0]):
                continue

            if abs(bbox[0] - bbox[2]) < 0.5:
                mid_b0 = (bbox[0] + bbox[2]) / 2
                bbox[0], bbox[2] = mid_b0, mid_b0
            if abs(bbox[1] - bbox[3]) < 0.5:
                mid_b0 = (bbox[1] + bbox[3]) / 2
                bbox[1], bbox[3] = mid_b0, mid_b0

            if bbox[0] - bbox[2] > 0:
                bbox[[0, 1]], bbox[[2, 3]] = bbox[[2, 3]], bbox[[0, 1]]
            if abs(bbox[0] - bbox[2]) <= 0 and bbox[1] > bbox[3]:
                bbox[[0, 1]], bbox[[2, 3]] = bbox[[2, 3]], bbox[[0, 1]]
            if bbox.min() < 0 or bbox.max() >= self.out_res[0]: # out of boundary
                if not self.point_rot_pro(bbox):
                    continue

            centerx, centery, bbox_use = self.decode(bbox)
            int_x = int(centerx) if centerx <= self.out_res[0] - 1 else self.out_res[0] - 1
            int_y = int(centery) if centery <= self.out_res[0] - 1 else self.out_res[0] - 1
            if center_hm[0][int_y][int_x] == 1:
                continue
            self.draw_center_point(int_x, int_y, center_hm, bbox_use, kp_displacement, line)
        center_mask[0] = np.where(line[0] > 0,  self.center_other_weight, center_mask[0])
        center_mask[0] = np.where(center_hm[0]> 0, self.centerweight, center_mask[0])
        kp_mask[0] = np.where(center_hm[0] > 0, 1., 0)
        line_mask[0] = np.where(line[0] > 0, self.lineweight, line_mask[0])

        ret = {"input": inp, 'center': center_hm, 'kp_displacement': kp_displacement, 'line': line,
               'kp_mask': kp_mask, 'center_mask': center_mask,
                'filename': file_name.split('.')[0],
               'line_mask': line_mask
               }
        return ret

class YorkDataset(data.Dataset):
    def __init__(self, data_dir, param):
        super(YorkDataset, self).__init__()
        self.param = param
        self.in_res = self.param.inres

        self.data_dir = data_dir
        self.images = os.listdir(self.data_dir)
        self.index = []
        minname_len = 1000
        maxname_len = 0
        for imname in self.images:
            if len(imname) > maxname_len:
                maxname_len = len(imname)
            if(len(imname)) < minname_len:
                minname_len = len(imname)
        if(minname_len) != maxname_len:
            for imname in self.images:
                imname = imname.rjust(maxname_len, '0')
                self.index.append(imname)
        else:
            self.index = self.images
        self.ordername = np.argsort(self.index)
        self.num_samples = len(self.images)

        print('Loaded {} samples'.format(self.num_samples))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        file_name = self.images[self.ordername[item]]
        img_path = self.data_dir + '/' + file_name
        img = cv.imread(img_path)
        inp = cv.resize(img, self.in_res)

        H, W, C = inp.shape
        hsv = cv.cvtColor(inp, cv.COLOR_BGR2HSV)
        imgv0 = hsv[..., 2]
        imgv = cv.resize(imgv0, (0, 0), fx=1. / 4, fy=1. / 4, interpolation=cv.INTER_LINEAR)
        imgv = cv.GaussianBlur(imgv, (5, 5), 3)
        imgv = cv.resize(imgv, (W, H), interpolation=cv.INTER_LINEAR)
        imgv = cv.GaussianBlur(imgv, (5, 5), 3)

        imgv1 = imgv0.astype(np.float32) - imgv + 127.5
        imgv1 = np.clip(imgv1, 0, 255).astype(np.uint8)
        hsv[..., 2] = imgv1
        inp = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

        inp = (inp.astype(np.float32) / 255.)
        inp = inp.transpose(2, 0, 1)

        ret = {"input": inp, 'filename': file_name.split('.')[0], 'origin_img': img}
        return ret
