import numpy as np
import scipy.io as sio
import os
import sys
import cv2 as cv
import glob

# prediction

'''
## lcnn
path = '/path-to-lcnn/net_output/lcnn_net_dir_output/*.npz'
save_path = './Lcnn/wire/'

os.makedirs(save_path, exist_ok=True)
mat_file = glob.glob(path)
for mat in mat_file:
    data = np.load(mat)['lines'].reshape(-1, 4)
    scores = np.load(mat)['score'].reshape(-1, 1)
    for i in range(len(data)):
        if i > 0 and (data[i] == data[0]).all():
            data = data[:i]
            scores = scores[:i]
            break
    print(data.shape)
    name = mat.split('/')[-1].split('.')[0]
    print(name)
    #img = cv.imread('/home/huangsiyu/evaluate/lcnn_data_prepare/valid-images/' + name + '.jpg')
    height, width = 320, 320
    f = open(save_path + '/' + name + '.txt', 'w')
    score = []
    num = 0
    for line in data:
        str_write = 'line ' + str(scores[num, 0]) + ' ' + str(line[1]/width * 128) + ' '+ str(line[0]/height*128) \
                    + ' '+ str(line[3]/width * 128) + ' ' +str(line[2]/height*128) + '\n'
        f.write(str_write)
        num+=1
    f.close()
sys.exit()
'''

import matplotlib.pyplot as plt

def imshow(im):
    sizes = im.shape
    height = float(sizes[0])
    width = float(sizes[1])
    img = im.copy()
    img[:,:,0] = im[:,:,2]
    img[:, :, 2] = im[:, :, 0]
    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.xlim([-0.5, sizes[1] - 0.5])
    plt.ylim([sizes[0] - 0.5, -0.5])
    plt.imshow(img)



# Other Result Usage
# path = '/AFM/atrous/scores_wire/*.mat' #save all line segments where aspect ratio in [0, 1]
# save_path = './AFM/wire' #-score
# Notice that AFM Need reverse score

# TP-LSD Result Path
path = '/TP-LSD-result-path/lmbd0.5/0.01/*.mat'
save_path = '/TP-LSD/wire'
wire_path = 'data/Wireframe/coco/images/val2017/'
# york_path = 'data/York/img/'
os.makedirs(save_path, exist_ok=True)
mat_file = glob.glob(path)

for mat in mat_file:
    data = sio.loadmat(mat)['lines']
    try:
        scores = sio.loadmat(mat)['score'].reshape(-1, 1)
    except:
        scores = sio.loadmat(mat)['scores'].reshape(-1, 1)
    print(data.shape)
    name = mat.split('/')[-1].split('.')[0]
    print(name)
    # height, width = 320, 320
    img = cv.imread(wire_path + name + '.png')
    # img = cv.imread(york_path + name + '.png')

    f = open(save_path + '/' + name + '.txt', 'w')
    score = []

    num = 0
    # plt.figure("LSD")
    # height, width = img.shape[:2]
    # img = cv.resize(img, (128, 128))
    # for line in data:
    #     str_write = 'line ' + str(scores[num, 0]) + ' ' + str(line[0] / width * 128) + ' ' + str(line[1] / height * 128) \
    #                 + ' ' + str(line[2] / width * 128) + ' ' + str(line[3] / height * 128) + '\n'
    #     cv.line(img, (int(line[0]/width * 128), int(line[1]/height*128)), (int(line[2]/width * 128), int(line[3]/height*128)), (0,255,0))
    #     f.write(str_write)
    #     num+=1
    # imshow(img)
    # plt.show()

    plt.figure("TP-LSD")
    img = cv.resize(img, (128, 128))
    height, width = 320, 320 # output img shape
    for line in data:
        str_write = 'line ' + str(scores[num, 0]) + ' ' + str(line[0] / width * 128) + ' ' + str(line[1] / height * 128) \
                    + ' ' + str(line[2] / width * 128) + ' ' + str(line[3] / height * 128) + '\n'
        # cv.line(img, (int(line[0] / width * 128), int(line[1] / height * 128)),
        #         (int(line[2] / width * 128), int(line[3] / height * 128)), (0, 255, 0))
        f.write(str_write)
        num += 1
    # imshow(img) # plot to ensure the right data
    # plt.show()
    f.close()
