import os
import argparse
import numpy as np
import sys
from modeling.Hourglass import HourglassNet
from modeling.TP_Net import Res160, Res320

class BasicParam(object):
    def __init__(self):
        # path for images
        self.dataset_dir = 'path for imgs'

        # path for results
        self.save_path = 'log/test/'

        self.batch_size = 1
        self.num_workers = 0
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
            raise Exception('cpu version for training is not implemented.')


''' Model choose
## TPLSD
self.model = Res320(self.head)
self.load_model_path = './pretraineds/Res320.pth'
self.inres = (320, 320)
self.outres = (320, 320)

## TPLSD-Lite
self.model = Res160(self.head)
self.load_model_path = './pretraineds/Res160.pth'
self.inres = (320, 320)
self.outres = (320, 320)

## TPLSD with 512 × 512 resolution
self.model = Res320(self.head)
self.load_model_path = './pretraineds/Res512.pth'
self.inres = (512, 512)
self.outres = (512, 512)

## Hourglass
self.model = HourglassNet(self.head)
self.load_model_path = './pretraineds/HG128.pth'
self.inres = (512, 512)
self.outres = (128, 128)
'''