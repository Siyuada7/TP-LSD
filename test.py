from progress.bar import Bar
from config.test_config import BasicParam
import os
import cv2 as cv
import time
import torch
import numpy as np

from dataset.dataset import YorkDataset
from progress.bar import Bar
from utils.utils import load_model
from utils.reconstruct import save_pic_mat, save_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cpu':
    raise Exception('cpu version for training is not implemented.')

print('Using device: ', device)


BasicParameters = BasicParam()
log_path = BasicParameters.save_path

model = BasicParameters.model
batch_size = BasicParameters.batch_size
num_workers = BasicParameters.num_workers

model = load_model(model, BasicParameters.load_model_path, BasicParameters.resume,
                   selftrain=BasicParameters.selftrain)
model = model.cuda()

test_dataset = YorkDataset(BasicParameters.dataset_dir, BasicParameters)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, num_workers=num_workers, pin_memory=True)

def inference(model, data_loader, threshold_list, lmbd_list, showvideo=True):
    if showvideo:
        win = "Line Detection"
        cv.namedWindow(win)
        cv.moveWindow(win, 60, 50)
    model.eval()
    with torch.no_grad():
        for lmbd in lmbd_list:
            for thresh in threshold_list:
                time_list = []
                num_iters = len(data_loader)
                bar = Bar('Threshold:{}'.format(thresh), max=num_iters)
                if not showvideo:
                    for iter_id, batch in enumerate(data_loader):
                        batch['input'] = batch['input'].float().cuda()
                        filename = batch['filename'][0]
                        torch.cuda.synchronize()
                        start_time = time.time()
                        outputs = model(batch['input'])

                        tmp_p, tmp_l, tmp_c, total_time = save_pic_mat(lmbd, thresh, outputs, batch['origin_img'], filename, log_path, save_mat=True)
                        mix_pic = np.concatenate([tmp_p, tmp_l, tmp_c], axis=1)
                        time_list.append(total_time - start_time)
                        save_dir = log_path + '/pic/' + str(thresh)
                        os.makedirs(save_dir, exist_ok=True)
                        cv.imwrite(save_dir + '/' +  batch['filename'][0] + '.png', mix_pic)
                        Bar.suffix = '[{0}/{1}]|'.format(iter_id, num_iters)
                        bar.next()
                    bar.finish()
                    total = sum(time_list)
                    print('Avg time per image: ', total/len(time_list))
                    print('FPS: ', 1/(total / len(time_list)))
                else:
                    for iter_id, batch in enumerate(data_loader):
                        batch['input'] = batch['input'].float().cuda()
                        torch.cuda.synchronize()
                        start_time = time.time()
                        outputs = model(batch['input'])
                        mix_pic, lines, total_time = save_image(lmbd, thresh, outputs, batch['origin_img'])
                        time_list.append(total_time - start_time)
                        total = sum(time_list)
                        fps = 1/(total / len(time_list))
                        cv.imshow(win, mix_pic)
                        cv.waitKey(1)
    if showvideo:
        cv.destroyAllWindows()

print('Starting testing...')

if not BasicParameters.showvideo:
    # threshold of the root-point detection confidence
    threshold_list=[0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    # power coefficient α in point filter module
    lmbd_list = [0.5]
    inference(model, test_loader, threshold_list, lmbd_list, showvideo=False)
else:
    threshold_list = [0.25]
    lmbd_list = [0.5]
    assert len(threshold_list) == 1 and len(lmbd_list) == 1
    inference(model, test_loader, threshold_list, lmbd_list, showvideo=True)
