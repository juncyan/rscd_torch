#调用官方库及第三方库
import torch
import numpy as np
#from tensorboardX import SummaryWriter
import datetime
from torch import optim
import os

#基础功能
from .val import evaluation
# from .predict import test
from torch.optim.lr_scheduler import StepLR


def train(obj):

    obj.logger.info("start train")
    model = obj.model
    
    # optimizer = optim.Adam(model.parameters(),lr= obj.args.lr, betas=(0.9, 0.999))
    # optimizer = optim.AdamW(model.parameters(), lr=obj.args.lr, weight_decay=5e-4)
    # max_itr = obj.args.iters * obj.traindata_num
    # lr_step = StepLR(optimizer, step_size=max_itr, gamma=0.5)
    

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=obj.args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)
    lr_step = StepLR(optimizer, 1, gamma=0.95, last_epoch=-1)

    max_miou = 0.
    best_iter = 0
    #early_stopping = Early_stopping(eps=2e-5,llen=10)
    #criterion = SegmentationLosses(weight=None,cuda=True).build_loss("ce")

    for epoch in range(obj.args.iters):
        now = datetime.datetime.now()
        model.train()
        loss_record = []

        for _,(image1, image2, label1, label2) in enumerate(obj.train_loader):

            image1 = image1.cuda(obj.device)
            image2 = image2.cuda(obj.device)
            # labels_bn = (label1>0).unsqueeze(1).cuda().float()
            label1 = label1.cuda(obj.device)
            label2 = label2.cuda(obj.device)
            
            out_change, outputs_A, outputs_B = model(image1, image2)
            
            if hasattr(model, "loss"):
                reduced_loss = model.loss(out_change, outputs_A, outputs_B, label1, label2, obj.device)
            
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
 
    # test(obj)
 




if __name__ == "__main__":
    print("work.train run")




