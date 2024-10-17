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
import cv2
import numpy as np
import torch
import pandas as pd
import glob
from torch.utils.data import DataLoader
import datetime 
from thop import profile
from .metric import Metrics
from .logger import load_logger


def predict(model, dataset, weight_path=None, data_name="test", num_classes=2, device=0):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        dataset (torch.io.DataLoader): Used to read and process test datasets.
        weights_path (string, optional): weights saved local.
    """


    if weight_path:
        layer_state_dict = torch.load(f"{weight_path}")
        model.load_state_dict(layer_state_dict)
    else:
        exit()

    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
    model_name = model.__str__().split("(")[0]

    img_dir = f"/mnt/data/Results/{data_name}/{model_name}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    color_label = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {model_name} on {data_name}")
    model = model.cuda(device)
    
    
    loader = DataLoader(dataset=dataset, batch_size=4, num_workers=0,
                                  shuffle=True, drop_last=True)

    evaluator = Metrics(num_class=num_classes)
    with torch.no_grad():
        for _, (img1, img2, label, name) in enumerate(loader):

            label = label
            img1 = img1.cuda(device)
            img2 = img2.cuda(device)
           
            pred = model(img1, img2)
           
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            elif hasattr(model, "prediction"):
                pred = model.prediction(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[-1]

            if pred.shape[1] > 1:
                pred = torch.argmax(pred, axis=1)
            pred = pred.squeeze().cpu()

            if label.shape[1] > 1:
                label = torch.argmax(label, axis=1)
            label = label.squeeze()
            label = label.numpy()

            evaluator.add_batch(pred, label)

            for idx, ipred in enumerate(pred):
                ipred = ipred.numpy()
                if (np.max(ipred) != np.min(ipred)):
                    flag = (label[idx] - ipred)
                    ipred[flag == -1] = 2
                    ipred[flag == 1] = 3
                    img = color_label[ipred]
                    cv2.imwrite(f"{img_dir}/{name[idx]}", img)

    evaluator.calc()
    miou = evaluator.Mean_Intersection_over_Union()
    acc = evaluator.Pixel_Accuracy()
    class_iou = evaluator.Intersection_over_Union()
    class_precision = evaluator.Class_Precision()
    kappa = evaluator.Kappa()
    m_dice = evaluator.Mean_Dice()
    f1 = evaluator.F1_score()
    macro_f1 = evaluator.Macro_F1()
    class_recall = evaluator.Recall()

    infor = "[PREDICT] #Images: {}".format(len(dataset))
    logger.info(infor)
    infor = "[METRICS] mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, mDice: {:.4f}, Macro_F1: {:.4f}".format(
            miou, acc, kappa, m_dice, macro_f1)
    logger.info(infor)

    logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
    logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
    logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
    logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))
    # print(batch_cost, reader_cost)

    _,c,w,h = img1.shape
    x= torch.rand([1,c,w,h]).cuda(device)
    flops, params = profile(model, [x,x])
    logger.info(f"[PREDICT] model flops is {int(flops)}, params is {int(params)}")
      
    

def test(obj=None):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        dataset (torch.io.DataLoader): Used to read and process test datasets.
        weights_path (string, optional): weights saved local.
    """
    assert obj != None, "obj is None, please check!"

    model = obj.model
    if obj.best_model_path:
        layer_state_dict = torch.load(f"{obj.best_model_path}")
        model.load_state_dict(layer_state_dict)
    else:
        exit()
    
    time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")

    img_dir = f"/mnt/data/Results/{obj.args.dataset}/{obj.model_name}_{time_flag}"
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)

    logger = load_logger(f"{img_dir}/prediction.log")
    logger.info(f"test {obj.args.dataset} on {obj.model_name}")
   
    color_label = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

    evaluator = Metrics(num_class=obj.args.num_classes)

    with torch.no_grad():
        for _, (img1, img2, label, name) in enumerate(obj.test_loader):
    
            label = label
            img1 = img1.cuda(obj.device)
            img2 = img2.cuda(obj.device)
           
            pred = model(img1, img2)
            
            if hasattr(model, "predict"):
                pred = model.predict(pred)
            elif hasattr(model, "prediction"):
                pred = model.prediction(pred)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[0]

            if pred.shape[1] > 1:
                pred = torch.argmax(pred, axis=1)
            pred = pred.squeeze().cpu()

            if label.shape[1] > 1:
                label = torch.argmax(label, axis=1)
            label = label.squeeze()
            label = label.numpy()

            evaluator.add_batch(pred, label)

            for idx, ipred in enumerate(pred):
                ipred = ipred.numpy()
                if (np.max(ipred) != np.min(ipred)):
                    flag = (label[idx] - ipred)
                    ipred[flag == -1] = 2
                    ipred[flag == 1] = 3
                    img = color_label[ipred]
                    cv2.imwrite(f"{img_dir}/{name[idx]}", img)

    evaluator.calc()
    miou = evaluator.Mean_Intersection_over_Union()
    acc = evaluator.Pixel_Accuracy()
    class_iou = evaluator.Intersection_over_Union()
    class_precision = evaluator.Class_Precision()
    kappa = evaluator.Kappa()
    m_dice = evaluator.Mean_Dice()
    f1 = evaluator.F1_score()
    macro_f1 = evaluator.Macro_F1()
    class_recall = evaluator.Recall()

    infor = "[PREDICT] #Images: {}".format(obj.test_num)
    obj.logger.info(infor)
    infor = "[METRICS] mIoU: {:.4f}, Acc: {:.4f}, Kappa: {:.4f}, mDice: {:.4f}, Macro_F1: {:.4f}".format(
            miou, acc, kappa, m_dice, macro_f1)
    obj.logger.info(infor)

    obj.logger.info("[METRICS] Class IoU: " + str(np.round(class_iou, 4)))
    obj.logger.info("[METRICS] Class Precision: " + str(np.round(class_precision, 4)))
    obj.logger.info("[METRICS] Class Recall: " + str(np.round(class_recall, 4)))
    obj.logger.info("[METRICS] Class F1: " + str(np.round(f1, 4)))
    # print(batch_cost, reader_cost)

    _,c,w,h = img1.shape
    x= torch.rand([1,c,w,h]).cuda(obj.device)
    flops, params = profile(model, [x,x])
    
    logger.info(f"[PREDICT] model flops is {int(flops)}, params is {int(params)}")

    img_files = glob.glob(os.path.join(img_dir, '*.png'))
    data = []
    for img_path in img_files:
        img = cv2.imread(img_path)
        lab = cls_count(img)
        # lab = np.argmax(lab, -1)
        data.append(lab)
    if data != []:
        data = np.array(data)
        pd.DataFrame(data).to_csv(os.path.join(img_dir, f'{obj.model_name}_violin.csv'), header=['TN', 'TP', 'FP', 'FN'], index=False)


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