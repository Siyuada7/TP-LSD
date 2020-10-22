# TP-LSD
Official implementation of paper "TP-LSD: Tri-points based line segment detector".

## Introduction
This demo uses TP-LSD to detect line segments in an image. The repo contains two core files: (1) PyTorch weight files and (2) The Network used in TP-LSD. 

Full paper PDF: [TP-LSD: Tri-Points Based Line Segment Detector](https://arxiv.org/abs/2009.05505)

Presentation PDF: [Poster at ECCV 2020](Paper/ECCV-tplsd.pdf)

Authors: Siyu Huang, Fangbo Qin, Pengfei Xiong, Ning Ding, Yijia He, Xiao Liu


Here are the line detection and tracking results running on the ICL-NUIM dataset.

<img src="Paper/processed_icl.gif" width="360">

## Dependencies
1. DCNV2: See modeling/DCNv2/README.md
2. lbdmod: See lbdmod/README.md
3. pytorch ≥ 1.0
4. opencv-python


## Running the Tracking Demo
This demo will run the TP-LSD network on an image sequence and extract line segments from the images.  
The tracks are formed by the `LineTracker` class which finds sequential pair-wise nearest neighbors using two-way matching of the lines' descriptors.
The demo script uses a helper class called `VideoStreamer` which can process inputs from three different input streams:

1. A directory of images, such as .png or .jpg
2. A video file, such as .mp4 or .avi
3. A USB Webcam

### Additional useful command line parameters
* Use `--method` to specify the method of line segments detection (default: lsd, option: lsd, edlines, tplsd).
* Use `--model` to specify choose the model to use (default: tp320, option: tp320, tplite, tp512, hg).
* Use `--skip` to skip the first few images if the input is movie or directory (default: 1).
* Use `--img_glob` to change the image file extension (default: *.png).
* Press the `q` key to quit.


### Run the demo on provided directory of images
`python demo_line.py imgdir_path --method=tplsd` 

### Run the demo on .mp4 file
`python demo_line.py xx.mp4 --method=tplsd`

### Run the demo via webcam (id #1) 
`python demo_line.py camera --camid=1 --method=tplsd`


## Running the Inference Demo
Prepare a directory of consequent images, such as .png or .jpg

### Hyper-parameter configurations
The basic configuration files for testing are saved in the `config/test_config.py`.

### Inference with pretrained models
The pretrained models for TP-LSD-Res34(320), TP-LSD-Res34(512), TP-LSD-Lite(320), TP-LSD-HG(512) can be downloaded from this link. Please place the weights into `<TP_LSD_root>/pretraineds`.
- For testing, please run the following commad, you could modify the model used and the path of test images in `config/test_config.yaml`.

- `python test.py`

## Metrics
### Line Matching Average Precision
LAP metrics is proposed based on line matching score (LMS), which measure the matching degree of line segments from a camera model perspective.
See README.md in Metrics/LAP.

### Pixel based Metric
The pixel based metric is to compare the detected line segment with the ground truth in a pixel-wise manner.
See README.md in Metrics/PixelBased.

### Structural Average Precision
sAP metrics uses the sum of squared error (LMS) between the predicted end-points and their ground truths as evaluation metric.
See README.md in Metrics/SAP.


