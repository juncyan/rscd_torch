import os
import math
import random
import numpy as np
import pandas as pd
from scipy import stats


class Metric_SCD(object):
    def __init__(self, num_class):
        self.__num_class = num_class
        self.__hist = np.zeros((self.__num_class,) * 2)
    
    def Get_Metric(self):
        hist_fg = self.__hist[1:, 1:]

        c2hist = np.zeros((2, 2))
        c2hist[0][0] = self.__hist[0][0]
        c2hist[0][1] = self.__hist.sum(1)[0] - self.__hist[0][0]
        c2hist[1][0] = self.__hist.sum(0)[0] - self.__hist[0][0]
        c2hist[1][1] = hist_fg.sum()
        hist_n0 = self.__hist.copy()
        hist_n0[0][0] = 0
        _, kappa_n0 = self.cal_kappa(hist_n0)
        iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
        IoU_fg = iu[1]
        IoU_mean = 0.5*(iu[0] + iu[1])
        Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
        
        po, kappa = self.cal_kappa(c2hist)
        pixel_sum = self.__hist.sum()
        change_pred_sum  = pixel_sum - self.__hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.__hist.sum(0)[0].sum()
        SC_TP = np.diag(hist_fg).sum()
        SC_Precision = SC_TP/change_pred_sum
        SC_Recall = SC_TP/change_label_sum
        Fscd = 2*(SC_Precision*SC_Recall) / (SC_Precision + SC_Recall)

        metrics = {
            'miou': IoU_mean,
            'kappa': kappa,
            'f1': Fscd,
            'sek': Sek,
            'pa': po,
            'prec': SC_Precision,
            'recall': SC_Recall}
           
        return metrics
    
    def get_hist(self, save_path=None):
        hist = pd.DataFrame(self.__hist)
        if save_path is None:
            return hist
        hist.to_csv(save_path, index=False, header=False, float_format='%.4f')
        return None
    
    def cal_kappa(self, hist):
        if hist.sum() == 0:
            po = 0
            pe = 1
            kappa = 0
        else:
            po = np.diag(hist).sum() / hist.sum()
            pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
            if pe == 1:
                kappa = 0
            else:
                kappa = (po - pe) / (1 - pe)
        return po, kappa

    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.__num_class)
        label = self.__num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.__num_class ** 2)
        confusion_matrix = count.reshape(self.__num_class, self.__num_class)
        return confusion_matrix

    # def add_batch(self, gt_image, pre_image):
    def add_batch(self, pred, lab):
        pred = np.array(pred)
        lab = np.array(lab)
        if len(lab.shape) == 4 and lab.shape[1] != 1:
            lab = np.argmax(lab, axis=1)

        if len(pred.shape) == 4 and pred.shape[1] != 1:
            pred = np.argmax(pred, axis=1)

        gt_image = np.squeeze(lab)
        pre_image = np.squeeze(pred)
        
        assert (np.max(pre_image) <= self.__num_class)
        assert gt_image.shape == pre_image.shape
        self.__hist += self.__generate_matrix(gt_image, pre_image)

    def reset(self):
        self.__hist = np.zeros((self.__num_class,) * 2)
        

class Metric(object):
    def __init__(self, num_class):
        self.__num_class = num_class
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)
        
        self.__TP = 0.#np.diag(self.__confusion_matrix)
        self.__RealN = 0.#np.sum(self.__confusion_matrix, axis=0)  # TP+FN
        self.__RealP = 0.#np.sum(self.__confusion_matrix, axis=1)  # TP+FP
        self.__sum = 0.#np.sum(self.__confusion_matrix)

    def Pixel_Accuracy(self):
        Acc = self.__TP.sum() / self.__sum
        return Acc

    def Class_Precision(self):
        #TP/TP+FP
        precision = self.__TP / (self.__RealP + 1e-5)
        # Acc = np.nanmean(Acc)
        return precision

    def Intersection_over_Union(self):
        IoU = self.__TP / (1e-5 + self.__RealP + self.__RealN - self.__TP)
        return IoU

    def Mean_Intersection_over_Union(self):
        IoU = self.Intersection_over_Union()
        MIoU = np.nanmean(IoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = self.__RealP / self.__sum
        iu = self.__TP / (1e-5 + self.__RealP + self.__RealN - self.__TP)
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Kappa(self):
        P0 = self.Pixel_Accuracy()
        Pe = np.sum(self.__RealP * self.__RealN) / (self.__sum * self.__sum)
        return (P0 - Pe) / (1 - Pe + 1e-5)
    
    def Kappa_n0(self):
        hist = self.__confusion_matrix.copy()
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        kappa = (po - pe) / (1 - pe + 1e-5)
        return kappa

    def F1_score(self, belta=1):
        precision = self.Class_Precision()
        recall = self.Recall()
        f1_score = (1 + belta * belta) * precision * recall / (belta * belta * precision + recall + 1e-5)
        return f1_score

    def Macro_F1(self, belta=1):
        return np.nanmean(self.F1_score(belta))

    def Dice(self):
        dice = 2 * self.__TP / (self.__RealN + self.__RealP + 1e-5)
        return dice

    def Mean_Dice(self):
        dice = self.Dice()
        return np.nanmean(dice)

    def Recall(self):  # 预测为正确的像素中确认为正确像素的个数
        #TP/ TP+FN
        recall = self.__TP / (self.__RealN + 1e-5)
        return recall
    
    def Get_Metric(self):
        self.calc()
        pa = np.round(self.Pixel_Accuracy(),4)
        iou = np.round(self.Intersection_over_Union(),4)
        miou = np.round(np.nanmean(iou),4)
        prices = np.round(self.Class_Precision(),4)
        f1 = np.round(self.F1_score(),4)
        mf1 = np.round(np.nanmean(f1),4)
        recall = np.round(self.Recall(),4)
        Pe = np.round(np.sum(self.__RealP * self.__RealN) / (self.__sum * self.__sum),4)
        kappa =  np.round((pa - Pe) / (1 - Pe),4)

        cls_iou = dict(zip(['iou_'+str(i) for i in range(self.__num_class)], iou))
        cls_precision = dict(zip(['precision_'+str(i) for i in range(self.__num_class)], prices))
        cls_recall = dict(zip(['recall_'+str(i) for i in range(self.__num_class)], recall))
        cls_F1 = dict(zip(['F1_'+str(i) for i in range(self.__num_class)], f1))

        metrics ={"pa":pa, "miou": miou, "mf1":mf1, "kappa":kappa}
        metrics.update(cls_iou)
        metrics.update(cls_F1)
        metrics.update(cls_precision)
        metrics.update(cls_recall)
        self.reset()
        return metrics

    def __generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.__num_class)
        label = self.__num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.__num_class ** 2)
        confusion_matrix = count.reshape(self.__num_class, self.__num_class)
        return confusion_matrix

    # def add_batch(self, gt_image, pre_image):
    def add_batch(self, pred, lab):
        pred = np.array(pred)
        lab = np.array(lab)
        # print(pred.shape, lab.shape)
        if len(lab.shape) == 4 and lab.shape[1] != 1:
            lab = np.argmax(lab, axis=1)

        if len(pred.shape) == 4 and pred.shape[1] != 1:
            pred = np.argmax(pred, axis=1)

        gt_image = np.array(np.squeeze(lab),dtype=np.uint8)
        pre_image = np.array(np.squeeze(pred),dtype=np.uint8)
        
        assert (np.max(pre_image) <= self.__num_class)
        # assert (len(gt_image) == len(pre_image))
        # print(gt_image.shape)
        # print(pre_image.shape)
        assert gt_image.shape == pre_image.shape
        self.__confusion_matrix += self.__generate_matrix(gt_image, pre_image)
        
    def calc(self):
        self.__TP = np.diag(self.__confusion_matrix)
        self.__RealN = np.sum(self.__confusion_matrix, axis=0)  # TP+FN
        self.__RealP = np.sum(self.__confusion_matrix, axis=1)  # TP+FP
        self.__sum = np.sum(self.__confusion_matrix)

    def reset(self):
        self.__confusion_matrix = np.zeros((self.__num_class,) * 2)
        self.__TP = 0.     #TP
        self.__RealN = 0.  # TP+FN
        self.__RealP = 0.  # TP+FP
        self.__sum = 0.  # np.sum(self.__confusion_matrix)
