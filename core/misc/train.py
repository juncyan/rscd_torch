#调用官方库及第三方库
import torch
import numpy as np
#from tensorboardX import SummaryWriter
import datetime
from torch import optim
import os

#基础功能
from .utils import get_params
from .val import evaluation
from .predict import test
from .utils import get_scheduler

from cd_models.losses import dice_loss


def train(obj):

    obj.logger.info("start train")
    model = obj.model
    #optimizer = optim.SGD(TNet.parameters(), opt_params)
    # optimizer = optim.SGD(
    #     params=[
    #         {'params': get_params(model, key='1x'), 'lr': 5e-4},
    #         {'params': get_params(model, key='10x'), 'lr': 5e-3}
    #     ],
    #     momentum = 0.9 , weight_decay = 1e-4)
    optimizer = optim.Adam(model.parameters(),lr= obj.args.lr, betas=(0.9, 0.999))
    # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult = 1, eta_min = 0, last_epoch = -1, verbose = False)
    
    max_itr = obj.args.iters * obj.traindata_num

    lr_step = get_scheduler(optimizer, max_itr, 'step')
    max_miou = 0.
    best_iter = 0
    #early_stopping = Early_stopping(eps=2e-5,llen=10)
    #criterion = SegmentationLosses(weight=None,cuda=True).build_loss("ce")

    for epoch in range(obj.args.iters):
        now = datetime.datetime.now()
        model.train()
        loss_record = []

        for _,(image1, image2, label) in enumerate(obj.train_loader):

            #optimizer = adjust_lr(optimizer, epoch*iter, max_itr)
            
            image1 = image1.cuda(obj.device)
            image2 = image2.cuda(obj.device)
            label = label.cuda(obj.device)
            
            pred = model(image1, image2)
            
            label = torch.argmax(label, dim=1, keepdim=True)
            if hasattr(model, "loss"):
                reduced_loss = model.loss(pred, label)
            else:
                if (type(pred) == tuple) or (type(pred) == list):
                    pred = pred[0]
                reduced_loss = dice_loss(pred, label)

            optimizer.zero_grad()  # 梯度清零
            reduced_loss.backward()  # 计算梯度
            optimizer.step()
            lr_step.step()

            loss_record.append(reduced_loss.item())

        loss_tm = np.mean(loss_record)

        obj.logger.info("[TRAIN] iter:{}/{}, learning rate:{:.6}, loss:{:.6}".format(epoch+1, obj.args.iters, optimizer.param_groups[0]['lr'], loss_tm))
        miou_eval = evaluation(obj)
        
        if miou_eval > max(0.5, max_miou):
            torch.save(model.state_dict(), obj.best_model_path)
            max_miou = miou_eval
            best_iter = epoch+1
        obj.logger.info("[TRAIN] train time is {:.2f}, best iter {}, max mIoU {:.4f}".format((datetime.datetime.now() - now).total_seconds(), best_iter, max_miou))
 
    test(obj)
 




if __name__ == "__main__":
    print("work.train run")




