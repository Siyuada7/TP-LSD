from lib.LineBox import BoundingBox
from lib.LineBoxes import BoundingBoxes
from lib.Evaluator_line import *
from lib.utils import *
import os

def getBoundingBoxes(gt_path, pred_path):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = os.path.dirname(os.path.abspath(__file__))
    folderGT = os.path.join(currentPath, gt_path)#'groundtruths_york/new')
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            x = float(splitLine[1])  # confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (128, 128),
                BBType.GroundTruth,
                format='GT')
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath, pred_path)#'TP_F_M_0.5/new')
    print(folderDet)
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()

    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt", "")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0]  # class
            confidence = float(splitLine[1])  # confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(
                nameOfImage,
                idClass,
                x,
                y,
                w,
                h,
                CoordinatesType.Absolute, (128, 128),
                BBType.Detected,
                confidence,
                format='GT')
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    return allBoundingBoxes

wire_gt = 'groundtruths_wire_line'
york_gt = 'groundtruths_york_line'
save_all = 'result/'
os.makedirs(save_all, exist_ok=True)

def cal_metric(gt_path, pred_path, save_file, thres):
    boundingboxes = getBoundingBoxes(gt_path, pred_path)
    save_file = save_all + save_file
    save_path = save_all + save_file.split('.')[0] + '.png'
    evaluator = Evaluator(save_file)
    evaluator.PlotPrecisionRecallCurve(
        boundingboxes,  # Object containing all bounding boxes (ground truths and detections)
        Threshold=thres,  # LMS threshold
        method=MethodAveragePrecision.EveryPointInterpolation,
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=True,
        savePath = save_path)  # Plot the interpolated precision curve
    metricsPerClass = evaluator.results
    print("Average precision values per class:\n")
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        average_precision = mc['AP']
        # Print AP per class
        print('%s: %f' % (c, average_precision))


Thress = [0.5]
for thres in Thress:
    wire_path = 'TP-LSD/wire'  # path for result
    save_file = 'TP-LSD-wire-' + str(thres) + '.npz'
    cal_metric(wire_gt, wire_path, save_file, thres)
