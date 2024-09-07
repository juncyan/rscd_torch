import torch
import numpy as np
from work.utils import colour_code_segmentation
import cv2
from datetime import datetime
import os
from .metrics import Metrics
import pandas as pd


def evaluation(obj):
    model = obj.model
    evaluator = Metrics(num_class=obj.args.num_classes)

    with torch.no_grad():
        model.eval()
        p_start = datetime.now()
        for _,(image1, image2, label) in enumerate(obj.val_loader):
            image1 = image1.cuda(obj.device)
            image2 = image2.cuda(obj.device)
            label = label

            pred = model(image1, image2)

            if hasattr(model, "predict"):
                pred = model.predict(pred)
            elif hasattr(model, "prediction"):
                pred = model.prediction(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[0]
           
            evaluator.add_batch(pred.cpu(), label)

    metrics = evaluator.Get_Metric()
    pa = metrics["pa"]
    miou = metrics["miou"]
    mf1 = metrics["mf1"]
    kappa = metrics["kappa"]

    if obj.logger != None:
        obj.logger.info("[EVAL] evalution {} images, time: {}".format(obj.val_num, datetime.now() - p_start))
        obj.logger.info("[METRICS] PA:{:.4},mIoU:{:.4},kappa:{:.4},Macro_f1:{:.4}".format(pa,miou,kappa,mf1))
        
    d = pd.DataFrame([metrics])
    if os.path.exists(obj.metric_path):
        d.to_csv(obj.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(obj.metric_path, index=False,float_format="%.4f")
    return miou


if __name__=="__main__":
    print("work.eval run")

    #TNet = torch.load(args.model_dir)