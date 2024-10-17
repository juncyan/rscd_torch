import torch
import torch.nn.functional as F
import numpy as np
from work.utils import colour_code_segmentation
import cv2
from datetime import datetime
import time
import os
from .metric import accuracy, SCDD_eval_all, AverageMeter
import pandas as pd


def evaluation(obj):
    model = obj.model
    torch.cuda.empty_cache()
   
    acc_meter = AverageMeter()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        model.eval()
        p_start = datetime.now()

        for _,(image1, image2, label1, label2,label, _) in enumerate(obj.test_loader):
            image1 = image1.cuda(obj.device)
            image2 = image2.cuda(obj.device)
            labels_A = np.array(label1, dtype=np.int64)
            labels_B = np.array(label2, dtype=np.int64)

            out_change, outputs_A, outputs_B = model(image1, image2)

            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask =  torch.argmax(out_change, axis=1).cpu().detach()

            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).cpu().numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).cpu().numpy()
    

            for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
                acc_A, valid_sum_A = accuracy(pred_A, label_A)
                acc_B, valid_sum_B = accuracy(pred_B, label_B)
                preds_all.append(pred_A)
                preds_all.append(pred_B)
                labels_all.append(label_A)
                labels_all.append(label_B)
                acc = (acc_A + acc_B)*0.5
                acc_meter.update(acc)
           
        kappa_n0, Fscd, MIoU, Sek = SCDD_eval_all(preds_all, labels_all, obj.args.num_classes)
        Acc = acc_meter.avg
        metrics = {"Sek":Sek,"Acc":Acc,"MIoU":MIoU,"Kappa":kappa_n0,"Fscd":Fscd}

    if obj.logger != None:
        obj.logger.info("[EVAL] evalution {} images, time: {}".format(obj.test_num, datetime.now() - p_start))
        obj.logger.info("[METRICS] Sek:{:.4},Acc:{:.4},MIoU:{:.4},Kappa:{:.4},Fscd:{:.4}".format(Sek,Acc,MIoU,kappa_n0,Fscd))
        
    d = pd.DataFrame([metrics])
    if os.path.exists(obj.metric_path):
        d.to_csv(obj.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(obj.metric_path, index=False,float_format="%.4f")
    return Sek


if __name__=="__main__":
    print("work.eval run")

    #TNet = torch.load(args.model_dir)