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

from cd_models.scd_sam.util.loss import CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity


def train(model, dataloader_train, dataloader_eval, dataloader_test, args):

    args.logger.info("start train")
    
    # optimizer = optim.Adam(model.parameters(),lr= args.lr, betas=(0.9, 0.999))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    max_itr = args.iters * args.traindata_num
    lr_step = StepLR(optimizer, step_size=max_itr, gamma=0.5)
    
    max_miou = 0.
    best_iter = 0
    
    criterion = CrossEntropyLoss2d()

    for epoch in range(args.iters):
        now = datetime.datetime.now()
        model.train()
        loss_record = []

        for image, label in tqdm(dataloader_train):

            image = image.to(args.device)
            label = label.to(args.device)

            # label = torch.argmax(label, 1)
            preds = model(image)
            
            optimizer.zero_grad()  
            reduced_loss = criterion(preds, label)
            
            
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




