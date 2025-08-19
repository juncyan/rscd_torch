import torch
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm
from ..cdmisc.utils import TimeAverager
from .metric import Metrics
import pandas as pd


def evaluation(model, dataloader_val, args):
    torch.cuda.empty_cache()
   
    evaluator = Metrics(num_class=args.num_classes)

    with torch.no_grad():
        model.eval()
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        batch_start = time.time()

        for image, label, _ in tqdm(dataloader_val):
            reader_cost_averager.record(time.time() - batch_start)

            image = image.to(args.device)
            label = np.array(label, dtype=np.int64)
            

            preds = model(image)

            batch_cost_averager.record(
                time.time() - batch_start, num_samples=len(preds))
            batch_cost = batch_cost_averager.get_average()
            reader_cost = reader_cost_averager.get_average()

            reader_cost_averager.reset()
            batch_cost_averager.reset()
            batch_start = time.time()

            preds = torch.argmax(preds, 1, False).cpu().numpy()
            evaluator.add_batch(preds, label)
        
    metrics, iou, f1 = evaluator.Get_Metric(True)
    miou = metrics['miou']

    if args.logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(args.test_num, batch_cost, reader_cost)
        args.logger.info(infor)
        args.logger.info("[METRICS] MIoU:{:.4}, Kappa:{:.4}, mF1:{:.4}, PA:{:.4}".format(miou,metrics['kappa'],metrics['mf1'], metrics['pa']))
        args.logger.info("[METRICS] Class IoU: " + str(np.round(iou, 4)))
        args.logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))
            
        
    d = pd.DataFrame([metrics])
    if os.path.exists(args.metric_path):
        d.to_csv(args.metric_path,mode='a', index=False, header=False,float_format="%.4f")
    else:
        d.to_csv(args.metric_path, index=False,float_format="%.4f")
    return miou


if __name__=="__main__":
    print("work.eval run")

    #TNet = torch.load(args.model_dir)