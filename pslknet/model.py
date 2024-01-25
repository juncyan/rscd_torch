import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ConvBn

from .modules import *


class PSLKNet_k9(nn.Module):
    #large kernel pseudo siamese network
    def __init__(self, in_channels=3, kernels=9):
        super().__init__()

        self.fa = PSBFA([64, 128, 256, 512])

        self.stage1 = STAF(in_channels, 64)#BFIB(2*in_channels, 64, kernels)
        self.stage2 = BFIB(64, 128, kernels)
        self.stage3 = BFIB(128, 256, kernels)
        self.stage4 = BFIB(256, 512, kernels)

        self.cbr1 = MF(128, 64)
        self.cbr2 = MF(256, 128)
        self.cbr3 = MF(512, 256)
        self.cbr4 = MF(1024, 512)

        self.up1 = UpBlock(512+128, 256)
        self.up2 = UpBlock(256+64, 64)
        # self.up3 = UpBlock(128+64, 64)

        self.cls1 = ConvBn(512, 2, 3)
        self.cls2 = ConvBn(512, 2, 3)
        self.classiier = nn.Sequential(nn.Conv2d(64, 2, 7, 1, 3), nn.BatchNorm2d(2), nn.Sigmoid())

    
    def forward(self, x1, x2):
        # x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        _, _, w, h = x1.shape
        a1, a2, a3, a4 = self.fa(x1, x2)

        f1 = self.stage1(x1, x2)
        m1 = self.cbr1(f1, a1)
        f2 = self.stage2(m1)
        m2 = self.cbr2(f2, a2)
        f3 = self.stage3(m2)
        m3 = self.cbr3(f3, a3)
        f4 = self.stage4(m3)
        m4 = self.cbr4(f4, a4)

        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        # print(a1.shape, a2.shape, a3.shape, a4.shape)
        
        r1 = self.up1(m4, m2)
        r2 = self.up2(r1, m1)
        # r3 = self.up3(r2, m1)

        l1 = self.cls1(m4)
        l1 = F.interpolate(l1, size=[w, h],mode='bilinear')

        l2 = self.cls2(r2)
        l2 = F.interpolate(l2, size=[w, h],mode='bilinear')

        y = F.interpolate(r2, size=[w, h],mode='bilinear')
        y = self.classiier(y)

        return y#, l1, l2
    
    # @staticmethod
    # def loss(pred, label, wdice=0.2):
    #     # label = paddle.argmax(label,axis=1)
    #     prob, l1, l2 = pred

    #     # label = paddle.argmax(label, 1).unsqueeze(1)
    #     label = paddle.to_tensor(label, paddle.float32)
        
    #     dsloss1 = nn.loss.BCELoss()(l1, label)
    #     dsloss2 = nn.loss.BCELoss()(l2, label)
    #     Dice_loss = 0.5*(dsloss1+dsloss2)

    #     label = paddle.argmax(label, 1).unsqueeze(1)
    #     # label = paddle.to_tensor(label, paddle.float16)

    #     CT_loss = nn.loss.CrossEntropyLoss(axis=1)(prob, label)
    #     CD_loss = CT_loss + wdice * Dice_loss
    #     return CD_loss
    
    # @staticmethod
    # def predict(pred):
    #     prob, _, _ = pred
    #     return prob

    
class SLKNet(nn.Module):
    #large kernel siamese network
    def __init__(self, in_channels=3, kernels=7):
        super().__init__()

        self.fa = SBFA([64, 128, 256, 512])

        self.stage1 = STAF(in_channels, 64)#BFIB(2*in_channels, 64, kernels)
        self.stage2 = BFIB(64, 128, kernels)
        self.stage3 = BFIB(128, 256, kernels)
        self.stage4 = BFIB(256, 512, kernels)

        # self.cls1 = layers.ConvBNAct(512, 2, 3, act_type="sigmoid")
        # self.cls2 = layers.ConvBNAct(512, 2, 3, act_type="sigmoid")
        self.cbr1 = MF(128,64)
        self.cbr2 = MF(256,128)
        self.cbr3 = MF(512,256)
        self.cbr4 = MF(1024,512)

        self.up1 = UpBlock(512+256, 256)
        self.up2 = UpBlock(256+128, 128)
        self.up3 = UpBlock(128+64, 64)

        self.classiier = nn.Sequential(nn.Conv2d(64, 2, 7, 1, 3), nn.BatchNorm2d(2), nn.Sigmoid())

    
    def forward(self, x1, x2):
        # x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        _, _, w, h = x1.shape
        a1, a2, a3, a4 = self.fa(x1, x2)

        f1 = self.stage1(x1, x2)
        m1 = self.cbr1(f1, a1)
        f2 = self.stage2(m1)
        m2 = self.cbr2(f2, a2)
        f3 = self.stage3(m2)
        m3 = self.cbr3(f3, a3)
        f4 = self.stage4(m3)
        m4 = self.cbr4(f4, a4)

        # print(f1.shape, f2.shape, f3.shape, f4.shape)
        # print(a1.shape, a2.shape, a3.shape, a4.shape)
        
        r1 = self.up1(m4, m3)
        r2 = self.up2(r1, m2)
        r3 = self.up3(r2, m1)

        # l1 = self.cls1(f4)
        # l1 = F.interpolate(l1, size=[w, h],mode='bilinear')

        # l2 = self.cls2(a4)
        # l2 = F.interpolate(l2, size=[w, h],mode='bilinear')

        y = F.interpolate(r3, size=[w, h],mode='bilinear')
        y = self.classiier(y)

        return y#, l1, l2
    
    # @staticmethod
    # def loss(pred, label, wdice=0.2):
    #     # label = paddle.argmax(label,axis=1)
    #     prob, l1, l2 = pred

    #     # label = paddle.argmax(label, 1).unsqueeze(1)
    #     label = paddle.to_tensor(label, paddle.float32)
        
    #     dsloss1 = nn.loss.BCELoss()(l1, label)
    #     dsloss2 = nn.loss.BCELoss()(l2, label)
    #     Dice_loss = 0.5*(dsloss1+dsloss2)

    #     label = paddle.argmax(label, 1).unsqueeze(1)
    #     # label = paddle.to_tensor(label, paddle.float16)

    #     CT_loss = nn.loss.CrossEntropyLoss(axis=1)(prob, label)
    #     CD_loss = CT_loss + wdice * Dice_loss
    #     return CD_loss
    
    # @staticmethod
    # def predict(pred):
    #     prob, _, _ = pred
    #     return prob