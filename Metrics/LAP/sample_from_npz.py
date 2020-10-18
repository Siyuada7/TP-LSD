from lib.Evaluator_line import *
from lib.utils import *
import matplotlib.pyplot as plt
import os

import numpy as np
import scipy.io as sio
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate

import sys
mpl.rcParams.update({"font.size": 12})
plt.rcParams["font.family"] = "Times New Roman"
del mpl.font_manager.weight_dict["roman"]
mpl.font_manager._rebuild()
plt.figure(figsize=(5, 4))

save_path = 'result/'

TP_LSD = 'result/TP-LSD-wire-0.5.npz'

path = [TP_LSD]
label = ['TP-LSD']
color = ['slateblue']


i=0
for p, l in zip(path, label):
    evaluator = Evaluator(None)
    evaluator.PlotPrecisionRecallCurveFromNPZ(
        p,  # Object containing all bounding boxes (ground truths and detections)
        method=MethodAveragePrecision.EveryPointInterpolation,  # As the official matlab code
        showAP=True,  # Show Average Precision in the title of the plot
        showInterpolatedPrecision=False,
        label=l, color=color[i])  # Plot the interpolated precision curve
    # Get metrics with PASCAL VOC metrics
    metricsPerClass = evaluator.results
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    i+= 1
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('%s: %f' % (c, average_precision))
f_scores = np.linspace(0.2, 0.8, num=8)
for f_score in f_scores:
    x = np.linspace(0.01, 1)
    y = f_score * x / (2 * x - f_score)
    l, = plt.plot(x[y >= 0], y[y >= 0], color="green", alpha=0.3)
    plt.annotate("f={0:0.1}".format(f_score), xy=(0.9, y[45] + 0.02), alpha=0.4)
plt.grid(True)

plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xticks(np.arange(0, 1.0, step=0.1))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.yticks(np.arange(0.0, 1.0, step=0.1))
plt.legend(loc=1)
#plt.title("PR Curve for Heatmap in Wireframe dataset")
plt.savefig(save_path + "/wire_lap.pdf", format="pdf", bbox_inches="tight")
plt.show()