import os
import argparse
import numpy as np
import sys
from modeling.Hourglass import HourglassNet
from modeling.TP_Net import Res160, Res320
from modeling.loss import weightedLoss

class BasicParam(object):
    def __init__(self):
        # path for images
        self.dataset_dir = '/media/uav514/AdaDisk/数据集/ICLdataset/ICL_NUIM/living_room_traj2_frei_png/rgb'
        self.dataset_dir = '/media/uav514/AdaDisk/数据集/Euroc_dataset/V1_01_easy/mav0/rect/cam0'

        # path for results
        self.save_path = 'log/testeur101/'

        self.batch_size = 1
        self.num_workers = 1
        self.head = {'center': 1, 'dis': 4, 'line': 1}
        self.cuda = True

        ## model && dataset
        self.resume = False
        self.selftrain = False

        # Replace it for other model. See details in below.
        self.model = Res320(self.head)
        self.load_model_path = './pretraineds/Res320.pth'
        self.inres = (320, 320)
        self.outres = (320, 320)

        self.logger = True

        self.showvideo = True

        os.makedirs(self.save_path, exist_ok=True)
        if self.cuda == False:
            print('cpu version for training is not implemented.')
            sys.exit()
        # set gpu devices
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.gpus
        # self.gpus = [i for i in range(len(self.gpus.split(',')))]
        # if len(self.gpus) <= 0:
        #     print('cpu version for training is not implemented.')
        #     sys.exit()
        # print("Set CUDA_VISIBLE_DEVICES to %s" % self.gpus)

''' Model choose
self.model = Res320(self.head)
self.load_model_path = './pretraineds/Res320.pth'
self.inres = (320, 320)
self.outres = (320, 320)

self.model = Res160(self.head)
self.load_model_path = './pretraineds/Res160.pth'
self.inres = (320, 320)
self.outres = (320, 320)

self.model = Res320(self.head)
self.load_model_path = './pretraineds/Res512.pth'
self.inres = (512, 512)
self.outres = (512, 512)

self.model = HourglassNet(self.head)
self.load_model_path = './pretraineds/HG128.pth'
self.inres = (512, 512)
self.outres = (128, 128)
'''