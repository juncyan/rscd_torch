#调用官方库及第三方库
import torch
import numpy as np
import datetime
from torch import optim
import os
from tqdm import tqdm
#基础功能
from .val import evaluation
from .predict import test
from torch.optim.lr_scheduler import StepLR
from .loss import loss, loss_lovasz
from cd_models.scd_sam.util.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity


def train(model, dataloader_train, dataloader_eval, dataloader_test, args):

    args.logger.info("start train")
    
    # optimizer = optim.Adam(model.parameters(),lr= args.lr, betas=(0.9, 0.999))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    max_itr = args.iters * args.traindata_num
    lr_step = StepLR(optimizer, step_size=max_itr, gamma=0.5)
    
    max_miou = 0.
    best_iter = 0
    #early_stopping = Early_stopping(eps=2e-5,llen=10)
    #criterion = SegmentationLosses(weight=None,cuda=True).build_loss("ce")

    seg_criterion = CrossEntropyLoss2d(ignore_index=0) 
    criterion_sc = ChangeSimilarity().to(args.device)

    for epoch in range(args.iters):
        now = datetime.datetime.now()
        model.train()
        loss_record = []

        for image1, image2, label1, label2, label in tqdm(dataloader_train):

            image1 = image1.to(args.device)
            image2 = image2.to(args.device)
            # labels_bn = (label1>0).unsqueeze(1).cuda().float()
            label1 = label1.to(args.device).long()
            label2 = label2.to(args.device).long()
            label = label.to(args.device)
            
            out_change, outputs_A, outputs_B = model(image1, image2)
            
            optimizer.zero_grad()  
            # print(out_change.shape, label.shape)
            # reduced_loss = loss_lovasz(out_change, outputs_A, outputs_B, label1, label2, label)
            loss_seg = seg_criterion(outputs_A, label1) * 0.5 +  seg_criterion(outputs_B, label2) * 0.5
            loss_sc = criterion_sc(outputs_A[:,1:], outputs_B[:,1:], label)
            loss_bn =  weighted_BCE_logits(out_change, label)
            
            reduced_loss = loss_seg + loss_bn + loss_sc
            
            reduced_loss.backward() 
            optimizer.step()
            lr_step.step()

            loss_record.append(reduced_loss.item())

        loss_tm = np.mean(loss_record)

        args.logger.info("[TRAIN] iter:{}/{}, learning rate:{:.6}, loss:{:.6}".format(epoch+1, args.iters, optimizer.param_groups[0]['lr'], loss_tm))
        miou = evaluation(model, dataloader_eval, args)
        
        if miou > max_miou:
            torch.save(model.state_dict(), args.best_model_path)
            max_miou = miou
            best_iter = epoch+1
        args.logger.info("[TRAIN] train time is {:.2f}, best iter {}, max MIoU {:.4f}".format((datetime.datetime.now() - now).total_seconds(), best_iter, max_miou))
 
    test(model, dataloader_test, args)
 




if __name__ == "__main__":
    print("work.train run")




