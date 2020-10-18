import os
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from .LineBox import *
from .LineBoxes import *
from .utils import *


from tqdm import tqdm
from .line_intersection import line_area_intersection

param_theta = 10
param_ratio = 0.5
z = 24

class Evaluator:
    def __init__(self, save_path):
        self.save_path = os.path.dirname(os.path.abspath(__file__)) + '/../' + save_path
    def GetMetricsFromNpz(self, filepath):
        ret = np.load(filepath, allow_pickle=True)['ret']
        return ret

    def GetLMSMetric(self,
                            boundingboxes,
                            Threshold=0.5,
                            method=MethodAveragePrecision.EveryPointInterpolation):
        ret = []  #
        # List with all ground truths
        groundTruths = []
        # List with all detections
        detections = []
        classes = []
        for bb in boundingboxes.getBoundingBoxes():
            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([
                    bb.getImageName(),
                    bb.getClassId(), 1,
                    bb.getAbsoluteBoundingBox_GT()
                ])
            else:
                detections.append([
                    bb.getImageName(),
                    bb.getClassId(),
                    bb.getConfidence(),
                    bb.getAbsoluteBoundingBox()
                ])
            # get class
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)
        # Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in classes:
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c
            gts = []
            [gts.append(g) for g in groundTruths if g[1] == c]
            npos = len(gts)
            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            # create dictionary with amount of gts for each image
            det = Counter([cc[0] for cc in gts])
            for key, val in det.items():
                det[key] = np.zeros(val)

            # Loop through detections
            for d in tqdm(range(len(dects))):
                # Find ground truth image
                gt = np.array([gt for gt in gts if gt[0] == dects[d][0]])
                LMSMax, jmax = Evaluator.LMS(dects[d][3], gt[:, 3])

                if LMSMax >= Threshold:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d] = 1  # count as true positive
                        det[dects[d][0]][jmax] = 1  # flag as already 'seen'
                        # print("TP")
                    else:
                        FP[d] = 1  # count as false positive
                        # print("FP")
                else:
                    FP[d] = 1  # count as false positive
                    # print("FP")
            # compute precision, recall and average precision

            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = sorted(acc_TP / npos)
            prec = sorted(np.divide(acc_TP, (acc_FP + acc_TP)))[::-1]
            # Depending on the method, call the right implementation
            if method == MethodAveragePrecision.EveryPointInterpolation:
                [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            else:
                [ap, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(rec, prec)
            # add class result in the dictionary to be returned
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP)
            }
            ret.append(r)
            np.savez(self.save_path, TP=TP, FP=FP, npos=npos, ret=ret)
        return ret

    def PlotPrecisionRecallCurve(self,
                                 boundingBoxes,
                                 Threshold=0.5,
                                 method=MethodAveragePrecision.EveryPointInterpolation,
                                 showAP=False,
                                 showInterpolatedPrecision=False,
                                 savePath=None,
                                 showGraphic=True):

        results = self.GetLMSMetric(boundingBoxes, Threshold, method)
        self.results = results
        # Each resut represents a class
        for result in results:
            if result is None:
                raise IOError('Error: Class %d could not be found.' % classId)

            classId = result['class']
            precision = result['precision']
            recall = result['recall']
            average_precision = result['AP']
            mpre = result['interpolated precision']
            mrec = result['interpolated recall']
            npos = result['total positives']
            total_tp = result['total TP']
            total_fp = result['total FP']

            plt.close()
            if showInterpolatedPrecision:
                if method == MethodAveragePrecision.EveryPointInterpolation:
                    plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
                elif method == MethodAveragePrecision.ElevenPointInterpolation:
                    # Uncomment the line below if you want to plot the area
                    # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
                    # Remove duplicates, getting only the highest precision of each recall value
                    nrec = []
                    nprec = []
                    for idx in range(len(mrec)):
                        r = mrec[idx]
                        if r not in nrec:
                            idxEq = np.argwhere(mrec == r)
                            nrec.append(r)
                            nprec.append(max([mpre[int(id)] for id in idxEq]))
                    plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
            plt.plot(recall, precision, label='Precision')
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.axis([0., 1., 0., 1.])
            plt.xticks(np.arange(0, 1., step=0.1))
            plt.yticks(np.arange(0, 1., step=0.1))
            if showAP:
                ap_str = "{0:.2f}%".format(average_precision * 100)
                # ap_str = "{0:.4f}%".format(average_precision * 100)
                plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(classId), ap_str))
            else:
                plt.title('Precision x Recall curve \nClass: %s' % str(classId))
            plt.legend(shadow=True)
            plt.grid()

            if savePath is not None:
                #plt.savefig(savePath)
                pass
            if showGraphic is True:
                plt.show()
                # plt.waitforbuttonpress()
                plt.pause(0.05)
        return results

    def PlotPrecisionRecallCurveFromNPZ(self,
                                 filepath,
                                 method=MethodAveragePrecision.EveryPointInterpolation,
                                 showAP=True,
                                 showInterpolatedPrecision=False,
                                 label=None,
                                 color = None,
                                 showGraphic=True):

        results = self.GetMetricsFromNpz(filepath)
        self.results = results
        result = None
        # Each resut represents a class
        for result in results:
            if result is None:
                raise IOError('Error: Class %d could not be found.' % classId)

            classId = result['class']
            precision = result['precision']
            recall = result['recall']
            average_precision = result['AP']
            mpre = result['interpolated precision']
            mrec = result['interpolated recall']
            npos = result['total positives']
            total_tp = result['total TP']
            total_fp = result['total FP']

            #plt.close()
            if showInterpolatedPrecision:
                if method == MethodAveragePrecision.EveryPointInterpolation:
                    plt.plot(mrec, mpre, '--r', label='Interpolated precision (every point)')
                elif method == MethodAveragePrecision.ElevenPointInterpolation:
                    # Uncomment the line below if you want to plot the area
                    # plt.plot(mrec, mpre, 'or', label='11-point interpolated precision')
                    # Remove duplicates, getting only the highest precision of each recall value
                    nrec = []
                    nprec = []
                    for idx in range(len(mrec)):
                        r = mrec[idx]
                        if r not in nrec:
                            idxEq = np.argwhere(mrec == r)
                            nrec.append(r)
                            nprec.append(max([mpre[int(id)] for id in idxEq]))
                    plt.plot(nrec, nprec, 'or', label='11-point interpolated precision')
            index1 = np.array(precision) > 0.1
            rrcs, pprs = [], []
            for iiii in range(len(index1)):
                if index1[iiii]:
                    rrcs.append(recall[iiii])
                    pprs.append(precision[iiii])
            if color:
                plt.plot(rrcs, pprs, label=label, color=color,  linewidth=2)
            else:
                plt.plot(rrcs, pprs, label=label, linewidth=2)
            plt.xlabel('recall')
            plt.ylabel('precision')
            plt.legend(shadow=True)
            plt.axis([0., 1., 0., 1.])
            plt.xticks(np.arange(0, 1., step=0.1))
            plt.yticks(np.arange(0, 1., step=0.1))
            plt.grid()
            #if savePath is not None:
                #plt.savefig(savePath)
            #if showGraphic is True:
                #plt.show()
                # plt.waitforbuttonpress()
                #plt.pause(0.05)
        return results

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        ii = []
        for i in range(len(mrec) - 1):
            if mrec[1:][i] != mrec[0:-1][i]:
                ii.append(i + 1)
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        # return [ap, mpre[1:len(mpre)-1], mrec[1:len(mpre)-1], ii]
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    # 11-point interpolated average precision
    def ElevenPointInterpolatedAP(rec, prec):
        # def CalculateAveragePrecision2(rec, prec):
        mrec = []
        # mrec.append(0)
        [mrec.append(e) for e in rec]
        # mrec.append(1)
        mpre = []
        # mpre.append(0)
        [mpre.append(e) for e in prec]
        # mpre.append(0)
        recallValues = np.linspace(0, 1, 11)
        recallValues = list(recallValues[::-1])
        rhoInterp = []
        recallValid = []
        # For each recallValues (0, 0.1, 0.2, ... , 1)
        for r in recallValues:
            # Obtain all recall values higher or equal than r
            argGreaterRecalls = np.argwhere(mrec[:] >= r)
            pmax = 0
            # If there are recalls above r
            if argGreaterRecalls.size != 0:
                pmax = max(mpre[argGreaterRecalls.min():])
            recallValid.append(r)
            rhoInterp.append(pmax)
        # By definition AP = sum(max(precision whose recall is above r))/11
        ap = sum(rhoInterp) / 11
        # Generating values for the plot
        rvals = []
        rvals.append(recallValid[0])
        [rvals.append(e) for e in recallValid]
        rvals.append(0)
        pvals = []
        pvals.append(0)
        [pvals.append(e) for e in rhoInterp]
        pvals.append(0)
        # rhoInterp = rhoInterp[::-1]
        cc = []
        for i in range(len(rvals)):
            p = (rvals[i], pvals[i - 1])
            if p not in cc:
                cc.append(p)
            p = (rvals[i], pvals[i])
            if p not in cc:
                cc.append(p)
        recallValues = [i[0] for i in cc]
        rhoInterp = [i[1] for i in cc]
        return [ap, rhoInterp, recallValues, None]

    @staticmethod
    def cal_norm_vector(use_bbox, z):
        one = np.ones([use_bbox.shape[0], 1])
        s = np.array([(use_bbox[:, 0]/z).reshape(-1, 1), (use_bbox[:, 1]/z).reshape(-1, 1), one])[:,:,0].transpose(1, 0)
        e= np.array([(use_bbox[:, 2]/z).reshape(-1, 1), (use_bbox[:, 3]/z).reshape(-1, 1), one])[:,:,0].transpose(1, 0)
        norm_vector = []
        for i in range(len(s)):
            tmp = np.cross(s[i], e[i])
            norm_vector.append(tmp / np.linalg.norm(tmp))
        return np.array(norm_vector)

    @staticmethod
    def angle_cal(gt_norm, pred_norm):
        angle_all = []
        for i in range(gt_norm.shape[0]):
            angle = np.arccos(np.clip(gt_norm[i].dot(pred_norm[i]), a_max=1, a_min=-1))*180/np.pi # 应该是没有绝对值的问题
            angle_all.append(angle)
        angle_all = np.array(angle_all)
        index = angle_all > 90
        angle_all[index] = 180 - angle_all[index]
        # if angle > 90:
        #     angle = 180 - angle
        return angle_all

    @staticmethod
    def LMS(boxA, boxBs_all):
        boxBs = np.array([boxBs_all[i]['pos'] for i in range(len(boxBs_all))])

        centers = np.array([boxBs_all[i]['center'] for i in range(len(boxBs_all))])
        gt_norm = np.array([boxBs_all[i]['norm'] for i in range(len(boxBs_all))]) # norm_vector
        # if boxes dont intersect
        size = len(boxBs)
        boxAs = np.expand_dims(boxA, 0).repeat(size, 0)
        use_boxA = np.array([(boxAs[:, 0] - centers[:, 0]), (boxAs[:, 1] - centers[:,1]),(boxAs[:,2] - centers[:, 0]), (boxAs[:,3] - centers[:,1])]).transpose(1, 0)
        #use_boxB = np.array([(boxBs[:,0] - centers[:, 0]), (boxBs[:,1] - centers[:,1]), (boxBs[:,2] - centers[:, 0]) , (boxBs[:,3] - centers[:,1])]).transpose(1, 0)
        #gt_norm = Evaluator.cal_norm_vector(use_boxB, z)
        pred_norm = Evaluator.cal_norm_vector(use_boxA, z)
        angle = Evaluator.angle_cal(gt_norm, pred_norm) # 两个法向量的夹角
        index = np.where(angle < param_theta)[0]
        if len(index) == 0:
            return 0, -1
        size = len(index)
        use_len_boxA = list(boxA)
        closet = 0
        closet_id = -1
        idx_valid, pd_covered = line_area_intersection(boxBs[index], boxA)
        for i in range(len(idx_valid)):
            if idx_valid[i] is False: # Angle condition
                continue
            l_A = np.sqrt((boxA[0]-boxA[2])**2 + (boxA[1]-boxA[3])**2)
            l_B = np.sqrt((boxBs[index[i],0]-boxBs[index[i],2])**2 + (boxBs[index[i],1]-boxBs[index[i],3])**2)
            tmp_ratio = pd_covered[i] / l_B
            if tmp_ratio < param_ratio: # Length condition
                continue
            s1 = np.array(boxA[2:4]) - np.array(boxA[0:2])
            s2 = np.array(boxBs[index[i], 2:4]) - np.array(boxBs[index[i], 0:2])
            theta = np.arccos(np.clip(s1.dot(s2)/ np.linalg.norm(s1)/np.linalg.norm(s2), a_max=1, a_min=-1))
            l_pred = abs(pd_covered[i]/ np.cos(theta))
            self_ratio = l_pred / l_A
            if self_ratio < param_ratio:
                continue
            use_ratio = np.clip((tmp_ratio + self_ratio) / 2, a_max=1, a_min=0)

            # LMF
            score_ang = 1 - angle[index[i]] / param_theta
            score_len = use_ratio
            iou = score_ang * score_len
            if iou > closet:
                closet = iou
                closet_id = i
        if closet_id == -1:
            return 0, -1

        assert closet >= 0
        return closet, index[closet_id]
