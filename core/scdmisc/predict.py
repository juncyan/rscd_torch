# Copyright (c) 2023 torchtorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import torch.nn.functional  as F
import pandas as pd
import glob
from skimage import io
import imageio.v3 as iio
import time
from torch.utils.data import DataLoader
import datetime 
from thop import profile
from tqdm import tqdm
from ..cdmisc.utils import TimeAverager
from .metric import Metric_SCD
from core.cdmisc.logger import load_logger


def predict(model, dataset, weight_path=None, data_name="test", num_classes=2, device=0):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        dataset (torch.io.DataLoader): Used to read and process test datasets.
        weights_path (string, optional): weights saved local.
    """

    if weight_path:
        checkpoint = torch.load(weight_path)
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    else:
        exit()

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
    model_name = model.__str__().split("(")[0]

    img_dir = f"/mnt/data/Results/{data_name}/{model_name}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    change_color = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {model_name} on {data_name}")
    model = model.cuda(device)
    
    test_num =dataset.__len__()
    label_info = np.transpose(dataset.label_info.values, [1,0])
    
    loader = DataLoader(dataset=dataset, batch_size=4, num_workers=0,
                                  shuffle=True, drop_last=True)

    evaluator = Metric_SCD(num_class=num_classes)

    with torch.no_grad():
        model.eval()
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        batch_start = time.time()

        for image1, image2, label1, label2, gt, file in tqdm(loader):
            reader_cost_averager.record(time.time() - batch_start)

            image1 = image1.to(device)
            image2 = image2.to(device)
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
    
            change_mask = F.sigmoid(out_change).cpu().detach()>0.5 #

            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).cpu().numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).cpu().numpy()

            evaluator.add_batch(preds_A, labels_A)
            evaluator.add_batch(preds_B, labels_B)
           
            for idx, (is1, is2, cdm) in enumerate(zip(preds_A, preds_B, change_mask)):
                cdm = np.array(cdm.squeeze(), np.uint8)
                if np.max(cdm) == np.min(cdm):
                    continue
                # flag_local = (gt[idx] - cdm)
                # cdm[flag_local == -1] = 2
                # cdm[flag_local == 1] = 3
                name = file[idx]
                # cdm = change_color[cdm]
                is1 = label_info[is1]
                is2 = label_info[is2]
                # iio.imsave(f"{img_dir}/{name}", np.uint8(cdm))
                fa = name.replace(".", "_A.")
                fb = name.replace(".", "_B.")
                iio.imwrite(f"{img_dir}/{fa}", np.uint8(is1))
                iio.imwrite(f"{img_dir}/{fb}", np.uint8(is2))

        

    evaluator.get_hist(save_path=f"{img_dir}/hist.csv")
    metrics = evaluator.Get_Metric()
    miou = metrics['miou']

    if logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(test_num, batch_cost, reader_cost)
        logger.info(infor)
        logger.info("[METRICS] MIoU:{:.4}, Kappa:{:.4}, F1:{:.4}, Sek:{:.4}".format(
            miou,metrics['kappa'],metrics['f1'],metrics['sek']))
        logger.info("[METRICS] PA:{:.4}, Prec.:{:.4}, Recall:{:.4}".format(
            metrics['pa'],metrics['prec'],metrics['recall']))
        
    _,c,w,h = image1.shape
    x= torch.rand([1,c,w,h]).cuda(device)
    flops, params = profile(model, [x,x])
    
    logger.info(f"[PREDICT] model flops is {int(flops)}, params is {int(params)}")
      
    return miou

    # evaluator.calc()
    # miou = evaluator.Mean_Intersection_over_Union()
    # acc = evaluator.Pixel_Accuracy()
    # class_iou = evaluator.Intersection_over_Union()
    # class_precision = evaluator.Class_Precision()
    # kappa = evaluator.Kappa()
    # m_dice = evaluator.Mean_Dice()
    # f1 = evaluator.F1_score()
    # macro_f1 = evaluator.Macro_F1()
    # class_recall = evaluator.Recall()

    # infor = "[PREDICT] #Images: {}".format(len(dataset))
    # logger.info(infor)
    # infor = "[METRICS] mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, mDice: {:.4f}, Macro_F1: {:.4f}".format(
    #         miou, acc, kappa, m_dice, macro_f1)
    # logger.info(infor)

    # logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
    # logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
    # logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
    # logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))
    # # print(batch_cost, reader_cost)

    # _,c,w,h = img1.shape
    # x= torch.rand([1,c,w,h]).cuda(device)
    # flops, params = profile(model, [x,x])
    # logger.info(f"[PREDICT] model flops is {int(flops)}, params is {int(params)}")
      

def test(model, dataloader_test, args):
    torch.cuda.empty_cache()
    if args.best_model_path:
        checkpoint = torch.load(f"{args.best_model_path}")
        model_dict = {}
        state_dict = model.state_dict()
        for k, v in checkpoint.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    else:
        exit()
    
    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")

    img_dir = f"/mnt/data/Results/{args.dataset}/{args.model}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {args.dataset} on {args.model}")
    change_color = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    evaluator = Metric_SCD(num_class=args.num_classes)

    with torch.no_grad():
        model.eval()
        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        batch_start = time.time()

        for image1, image2, label1, label2, gt, file in tqdm(dataloader_test):
            reader_cost_averager.record(time.time() - batch_start)

            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
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
            change_mask = F.sigmoid(out_change).cpu().detach()>0.5 #torch.argmax(out_change, axis=1).cpu().detach() #F.sigmoid(out_change).cpu().detach()>0.5 #
            change_mask = change_mask.squeeze()

            preds_A = torch.argmax(outputs_A, dim=1)
            preds_B = torch.argmax(outputs_B, dim=1)
            preds_A = (preds_A*change_mask.squeeze().long()).cpu().numpy()
            preds_B = (preds_B*change_mask.squeeze().long()).cpu().numpy()

            evaluator.add_batch(preds_A, labels_A)
            evaluator.add_batch(preds_B, labels_B)

            for idx, (is1, is2, cdm) in enumerate(zip(preds_A, preds_B, change_mask)):
                cdm = np.array(cdm.squeeze(), np.uint8)
                if np.max(cdm) == np.min(cdm):
                    continue
                # flag_local = (gt[idx] - cdm)
                # cdm[flag_local == -1] = 2
                # cdm[flag_local == 1] = 3
                name = file[idx]
                # cdm = change_color[cdm]
                is1 = args.label_info[is1]
                is2 = args.label_info[is2]
                # iio.imsave(f"{img_dir}/{name}", np.uint8(cdm))
                fa = name.replace(".", "_A.")
                fb = name.replace(".", "_B.")
                iio.imwrite(f"{img_dir}/{fa}", np.uint8(is1))
                iio.imwrite(f"{img_dir}/{fb}", np.uint8(is2))

    evaluator.get_hist(save_path=f"{img_dir}/hist.csv")
    metrics = evaluator.Get_Metric()
    miou = metrics['miou']

    if args.logger != None:
        infor = "[EVAL] Images: {} batch_cost {:.4f}, reader_cost {:.4f}".format(args.test_num, batch_cost, reader_cost)
        args.logger.info(infor)
        args.logger.info("[METRICS] MIoU:{:.4}, Kappa:{:.4}, F1:{:.4}, Sek:{:.4}".format(
            miou,metrics['kappa'],metrics['f1'],metrics['sek']))
        args.logger.info("[METRICS] PA:{:.4}, Prec.:{:.4}, Recall:{:.4}".format(
            metrics['pa'],metrics['prec'],metrics['recall']))
        
    _,c,w,h = image1.shape
    x= torch.rand([1,c,w,h]).cuda(args.device)
    flops, params = profile(model, [x,x])
    
    logger.info(f"[PREDICT] model flops is {int(flops)}, params is {int(params)}")
      
    return miou


def cls_count(label):
    cls_nums = []
    color_label = np.array([[0, 0, 0], [255, 255, 255], [0, 128, 0], [0, 0, 128]])
    for info in color_label:
        color = info
        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        matrix = np.sum(equality, axis=-1)
        nums = np.sum(matrix == 3)
        cls_nums.append(nums)
    return cls_nums