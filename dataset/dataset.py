import numpy as np
import os
import cv2 as cv
import torch.utils.data as data

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
