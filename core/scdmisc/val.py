import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
import os
from ..cdmisc.utils import TimeAverager
from .metric import Metrics
import pandas as pd


def evaluation(obj):
    model = obj.model
    torch.cuda.empty_cache()
   
    evaluator = Metrics(num_class=obj.num_classes)

    with torch.no_grad():
        model.eval()
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        batch_start = time.time()

        for _,(image1, image2, label1, label2,_, _) in enumerate(obj.test_loader):
            reader_cost_averager.record(time.time() - batch_start)

            image1 = image1.cuda(obj.device)
            image2 = image2.cuda(obj.device)
            labels_A = np.array(label1, dtype=np.int64)
            labels_B = np.array(label2, dtype=np.int64)

            out_change, outputs_A, outputs_B = model(image1, image2)

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(out_change))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

            outputs_A = outputs_A.cpu().detach()
            outputs_B = outputs_B.cpu().detach()
            change_mask =  torch.argmax(out_change, axis=1).cpu().detach()

            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).cpu().numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).cpu().numpy()

            evaluator.add_batch(preds_A, labels_A)
            evaluator.add_batch(preds_B, labels_B)
        
    metrics = evaluator.Get_Metric()
    miou = metrics['miou']

    if obj.logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(obj.val_num, batch_cost, reader_cost)
        obj.logger.info(infor)
        obj.logger.info("[METRICS] MIoU:{:.4}, Kappa:{:.4}, F1:{:.4}, Sek:{:.4}".format(
            miou,metrics['kappa'],metrics['f1'],metrics['sek']))
        obj.logger.info("[METRICS] PA:{:.4}, Prec.:{:.4}, Recall:{:.4}".format(
            metrics['pa'],metrics['prec'],metrics['recall']))
        
    d = pd.DataFrame([metrics])
    if os.path.exists(obj.metric_path):
        d.to_csv(obj.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(obj.metric_path, index=False,float_format="%.4f")
    return miou


if __name__=="__main__":
    print("work.eval run")

    #TNet = torch.load(args.model_dir)