#!/usr/bin/env python
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2018
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Daniel DeTone (ddetone)
#                       Tomasz Malisiewicz (tmalisiewicz)
#  Revision author: Siyu Huang
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

import argparse
import glob
import numpy as np
import os
import time
import cv2 as cv
from lbdmod.build import pylbd
import torch
from utils.reconstruct import TPS_line

myjet = np.array([[0.        , 0.        , 0.5       ],
                  [0.        , 0.        , 0.99910873],
                  [0.        , 0.37843137, 1.        ],
                  [0.        , 0.83333333, 1.        ],
                  [0.30044276, 1.        , 0.66729918],
                  [0.66729918, 1.        , 0.30044276],
                  [1.        , 0.90123457, 0.        ],
                  [1.        , 0.48002905, 0.        ],
                  [0.99910873, 0.07334786, 0.        ],
                  [0.5       , 0.        , 0.        ]])

class LineTracker(object):

  def __init__(self,max_num):
    self.maxnum = max_num
    self.point_list = []
    self.desc_list = []
    self.match_list = []
    # self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    self.matcher = cv.BFMatcher(cv.NORM_HAMMING)
    
  def update(self, pts, desc):
    if len(pts) < 1:
      return 

    if len(self.point_list)>self.maxnum + 1:
      self.point_list.pop()
      self.desc_list.pop()

    tmppts = []
    for p in pts:
      tmppts.append([p[0], p[1], p[2], p[3]])
    self.point_list.insert(0,tmppts)
    self.desc_list.insert(0,desc)

    if len(self.point_list) > 1:
      tmpmatches = self.matcher.knnMatch(self.desc_list[0], self.desc_list[1], k=2)
      matches = [m for m, n in tmpmatches if m.distance < 20 and m.distance < n.distance * 0.7]
      matches = sorted(matches, key=lambda x: x.distance)
      if len(self.match_list) > self.maxnum:
        self.match_list.pop()
      self.match_list.insert(0,matches)

  def draw_tracks(self, out, max_match):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    stroke = 1
    index_last = []
    # max_match = min(max_match, len(self.point_list))

    for i in range(len(self.match_list)):
      
      if i == 0:
        clr2 = (255, 0, 0)
        j = 0
        for index in self.match_list[i]:
          start = ((int(self.point_list[i][index.queryIdx][0]),int(self.point_list[i][index.queryIdx][1])))
          end = ((int(self.point_list[i][index.queryIdx][2]),int(self.point_list[i][index.queryIdx][3])))
          mid = (np.array(start) + np.array(end)) / 2
          cv.circle(out, (int(mid[0]), int(mid[1])), stroke, clr2, -1, lineType=16)
          cv.line(out, start, end, clr2, 2, lineType=16)
          index_last.append(index.queryIdx)
          j = j+1
          if j > max_match:
            break

      clr = myjet[i]*255
      index_next = []
      j = 0
      for index in self.match_list[i]:
        if index.queryIdx in index_last:
          start1 = ((int(self.point_list[i][index.queryIdx][0]),int(self.point_list[i][index.queryIdx][1])))
          end1 = ((int(self.point_list[i][index.queryIdx][2]),int(self.point_list[i][index.queryIdx][3])))
          start2 = ((int(self.point_list[i+1][index.trainIdx][0]),int(self.point_list[i+1][index.trainIdx][1])))
          end2 = ((int(self.point_list[i+1][index.trainIdx][2]),int(self.point_list[i+1][index.trainIdx][3])))
          p1 = (np.array(start1) + np.array(end1)) / 2
          p2 = (np.array(start2) + np.array(end2)) / 2
          index_next.append(index.trainIdx)
          cv.line(out, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), clr, thickness=stroke, lineType=16)
        
        j = j + 1
        if j > max_match:
          break
      
      index_last = index_next

    return out

class VideoStreamer(object):
  """ Class to help process image streams. Three types of possible inputs:"
    1.) USB Webcam.
    2.) A directory of images (files in directory matching 'img_glob').
    3.) A video file, such as an .mp4 or .avi file.
  """
  def __init__(self, basedir, camid, skip, img_glob):
    self.cap = []
    self.camera = False
    self.video_file = False
    self.listing = []
    self.i = 0
    self.skip = skip
    self.needsort = False
    # If the "basedir" string is the word camera, then use a webcam.
    if basedir == "camera/" or basedir == "camera":
      print('==> Processing Webcam Input.')
      self.cap = cv.VideoCapture(camid)
      self.listing = range(0, self.maxlen)
      self.camera = True
    else:
      # Try to open as a video.
      self.cap = cv.VideoCapture(basedir)
      lastbit = basedir[-4:len(basedir)]
      if (type(self.cap) == list or not self.cap.isOpened()) and (lastbit == '.mp4'):
        raise IOError('Cannot open movie file')
      elif type(self.cap) != list and self.cap.isOpened() and (lastbit != '.txt'):
        print('==> Processing Video Input.')
        num_frames = int(self.cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.listing = range(0, num_frames)
        self.listing = self.listing[::self.skip]
        self.camera = True
        self.video_file = True
        self.maxlen = len(self.listing)
      else:
        print('==> Processing Image Directory Input.')
        minname_len = 1000000
        maxname_len = 0
        self.index = []
        search = os.path.join(basedir, img_glob)
        self.listing = glob.glob(search)
        for imname in self.listing:
            name = imname.split('/')[-1]
            if len(name) > maxname_len:
                maxname_len = len(name)
            if(len(name)) < minname_len:
                minname_len = len(name)
        if(minname_len) != maxname_len:
            for imname in self.listing:
                name = imname.split('/')[-1]
                name = name.rjust(maxname_len, '0')
                self.index.append(name)
            self.needsort = True
        else:
            self.index = self.listing
        self.ordername = np.argsort(self.index)
        self.maxlen = len(self.ordername)
        if self.maxlen == 0:
          raise IOError('No images were found (maybe bad \'--img_glob\' parameter?)')

  def read_image(self, index):
    """ Read image as grayscale and resize to img_size.
    Inputs
      impath: Path to input image.
      img_size: (W, H) tuple specifying resize size.
    Returns
      grayim: float32 numpy array sized H x W with values in range [0, 1].
    """
    if self.needsort:
      impath = self.listing[self.ordername[index]]
    else:
      impath = self.listing[index]

    image = cv.imread(impath)
    grayim = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    if grayim is None:
      raise Exception('Error reading image %s' % impath)
    # Image is resized via opencv.
    # interp = cv.INTER_AREA
    return grayim, image

  def next_frame(self):
    """ Return the next frame, and increment internal counter.
    Returns
       image: Next H x W image.
       status: True or False depending whether image was loaded.
    """
    if self.i == self.maxlen:
      return (None, None, False)
    if self.camera:
      ret, image = self.cap.read()
      if ret is False:
        print('VideoStreamer: Cannot get image from camera (maybe bad --camid?)')
        return (None,None, False)
      if self.video_file:
        self.cap.set(cv.CAP_PROP_POS_FRAMES, self.listing[self.i])
      input_image = cv.resize(image, (self.sizer[1], self.sizer[0]),
                               interpolation=cv.INTER_AREA)
      input_image = cv.cvtColor(input_image, cv.COLOR_RGB2GRAY)
    else:
      # image_file = self.listing[self.i]
      input_image, image = self.read_image(self.i)
    # Increment internal counter.
    self.i = self.i + 1
    return (input_image, image, True)

class TplsdDetect:
  def __init__(self):
    from utils.utils import load_model
    from modeling.TP_Net import Res160, Res320
    from modeling.Hourglass import HourglassNet

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
      raise EOFError('cpu version for training is not implemented.')
    print('Using device: ', device)
    self.head = {'center': 1, 'dis': 4, 'line': 1}
    self.model = load_model(Res320(self.head), './pretraineds/Res320.pth', False, False)
    self.model = self.model.cuda().eval()
    self.in_res = (320, 320)

  def getlines(self, outputs, oriimg):
    output = outputs[-1]
    H, W = output['center'].shape[2:4]
    lines, start_point, end_point, pos, endtime = TPS_line(output, 0.25, 0.5, H, W)
    W_ = oriimg.shape[1] / W
    H_ = oriimg.shape[0] / H
    lines[:, [0, 2]] *= W_
    lines[:, [1, 3]] *= H_
    return lines

  def detect_tplsd(self, img):
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
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0).cuda()
    with torch.no_grad():
      outputs = self.model(inp)
    lines = self.getlines(outputs, img)
    return lines




if __name__ == '__main__':

  # Parse command line arguments.
  parser = argparse.ArgumentParser(description='Line Demo.')
  parser.add_argument('input', type=str, default='',
      help='Image directory or movie file or "camera" (for webcam).')
  parser.add_argument('--method', type=str, default='lsd',
                      help='Line detection method. (default: lsd, edlines, tplsd)')
  parser.add_argument('--camid', type=int, default=0,
      help='OpenCV webcam video capture ID, usually 0 or 1 (default: 0).')
  parser.add_argument('--img_glob', type=str, default='*.png',
      help='Glob match if directory of images is specified (default: \'*.png\').')
  parser.add_argument('--skip', type=int, default=1,
      help='Images to skip if input is movie or directory (default: 1).')
  parser.add_argument('--waitkey', type=int, default=1,
      help='OpenCV waitkey time in ms (default: 1).')
  opt = parser.parse_args()
  print(opt)


  print('==> Loading video.')
  # This class helps load input images from different sources.
  vs = VideoStreamer(opt.input, opt.camid, opt.skip, opt.img_glob)

  print('==> Successfully loaded video.')

  # This class helps merge consecutive point matches into tracks.
  tracker = LineTracker(5)
  print('==> Successfully loaded tracker model.')

  if opt.method == 'lsd':
    print('==> Detect Line Segments with LSD.')
  elif opt.method == 'edlines':
    print('==> Detect Line Segments with EdLines.')
  elif opt.method == 'tplsd':
    print('==> Detect Line Segments with TP-LSD.')
    tplsd = TplsdDetect()
  else:
    raise EOFError('Please specify the method of line segment detection.')

  # Create a window to display the demo.
  win = 'Line Tracker'
  cv.namedWindow(win)

  print('==> Running Demo.')

  t_begin = time.time()
  frame = 0
  while True:

    start = time.time()
    img, oriimg, status = vs.next_frame() # gray
    if status is False:
      break

    # Get points and descriptors.
    start1 = time.time()
    if opt.method == 'lsd':
      kls = pylbd.detect_lsd(img, 1, 1.44)
    elif opt.method == 'edlines':
      kls = pylbd.detect_edlines(img, 1, 1.44)
    elif opt.method == 'tplsd':
      kls = tplsd.detect_tplsd(oriimg)

    des = pylbd.describe_with_lbd(img, kls, 1, 1.44)

    tracker.update(kls, des)

    end1 = time.time()

    # Display visualization image to screen.
    out = oriimg
    tracker.draw_tracks(out,200)

    cv.imshow(win,out)
    key = cv.waitKey(opt.waitkey) & 0xFF
    if key == ord('q'):
      print('Quitting, \'q\' pressed.')
      break


    end = time.time()

    net_t = (1./ float(end1 - start))
    total_t = (1./ float(end - start))
    frame = frame + 1

  # Close any remaining windows.
  cv.destroyAllWindows()
  t_end = time.time()
  print("Total time spent:%f"%(t_end-t_begin))
  print("Average frame rate:%f"%(frame/(t_end-t_begin)))

  print('==> Finshed Demo.')
