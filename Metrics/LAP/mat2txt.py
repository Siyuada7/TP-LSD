import numpy as np
import scipy.io as sio
import os
import sys
import cv2 as cv
import glob

## Groundtruth
# img_path = 'data/York/images/'
# path = 'data/York/GT/*_line.mat'
# save_path = './groundtruths_york_line/'

img_path = 'data/Wirefeame/val2017/'
path = 'data/Wirefeame/linemat/*_line.mat'
save_path = './groundtruths_wire_line/'

os.makedirs(save_path, exist_ok=True)
mat_file = glob.glob(path)

for mat in mat_file:
    data = sio.loadmat(mat)['lines']
    # print(data.shape)
    name = mat.split('/')[-1].split('_')[0]
    print(name)
    img = cv.imread(img_path + name + '.png')
    height, width = img.shape[:2]

    f = open(save_path + '/' + name + '.txt', 'w')
    for line in data:
        str_write = 'line ' + str(line[0]/width * 128) + ' '+ str(line[1]/height*128) \
                    + ' '+ str(line[2]/width * 128) + ' ' +str(line[3]/height*128) + '\n'
        f.write(str_write)
    f.close()
sys.exit()
