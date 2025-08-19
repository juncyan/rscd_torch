import torch
import numpy as np
from datetime import datetime
import os
from tqdm import tqdm
from .metrics import Metrics
import pandas as pd


def evaluation(model,dataloader_eval,args):

    evaluator = Metrics(num_class=args.num_classes)
    with torch.no_grad():
        model.eval()
        p_start = datetime.now()
        num_eval = 0
        for image1, image2, label, _ in tqdm(dataloader_eval):
            num_eval +=1
            image1 = image1.cuda(args.device)
            image2 = image2.cuda(args.device)
            label = label

            pred = model(image1, image2)

            if hasattr(model, "predict"):
                pred = model.predict(pred)
            elif hasattr(model, "prediction"):
                pred = model.prediction(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[args.pred_idx]
            # pred = torch.where(torch.sigmoid(pred) > 0.5, 1, 0)
            # print(pred)
           
            evaluator.add_batch(pred.cpu(), label)

    metrics = evaluator.Get_Metric()
    pa = metrics["pa"]
    miou = metrics["miou"]
    mf1 = metrics["mf1"]
    kappa = metrics["kappa"]
    iou1 = metrics["iou_1"]

    if args.logger != None:
        args.logger.info("[EVAL] evalution {} images, time: {}".format(num_eval * args.batch_size, datetime.now() - p_start))
        args.logger.info("[METRICS] PA:{:.4},mIoU:{:.4},kappa:{:.4},Macro_f1:{:.4}, IoU 1:{:.4}".format(pa,miou,kappa,mf1, iou1))
        
    d = pd.DataFrame([metrics])
    if os.path.exists(args.metric_path):
        d.to_csv(args.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(args.metric_path, index=False,float_format="%.4f")
    return iou1


if __name__=="__main__":
    print("work.eval run")

    #TNet = torch.load(args.model_dir)