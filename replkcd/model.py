import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *


class LKPSNet(nn.Module):
    #large kernel convolution based pseudo siamese network
    def __init__(self, in_channels=3, kernels=9):
        super().__init__()

        self.fa = PSAA([64, 128, 256, 512])

        self.stage1 = STAF(in_channels, 64)
        self.stage2 = BFELKB(64, 128, kernels)
        self.stage3 = BFELKB(128, 256, kernels)
        self.stage4 = BFELKB(256, 512, kernels)

        self.cbr1 = MF(128, 64)
        self.cbr2 = MF(256, 128)
        self.cbr3 = MF(512, 256)
        self.cbr4 = MF(1024, 512)

        self.up1 = UpBlock(512+256, 256)
        self.up2 = UpBlock(256+128, 64)
        self.up3 = UpBlock(128+64, 64)

        # self.cls1 = nn.Sequential(ConvBn(512, 2, 3), nn.Sigmoid())
        # self.cls2 = nn.Sequential(ConvBn(256, 2, 3), nn.Sigmoid())
        # self.classiier = nn.Sequential(nn.Conv2d(64, 2, 7, 1, 3), nn.BatchNorm2d(2), nn.Sigmoid())

    
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

        r1 = self.up1(m4, m3)
        r2 = self.up2(r1, m2)
        r3 = self.up3(r2, m1)

        y = F.interpolate(r3, size=[w, h],mode='bilinear')
        y = self.classiier(y)

        return y
    
    # @staticmethod
    # def loss(pred, label, wdice=0.4):
    #     # label = torch.argmax(label,axis=1)
    #     prob, l1, l2 = pred

    #     # label = torch.argmax(label, 1).unsqueeze(1)
    #     label = torch.clone(label).detach().type(torch.float32)
        
    #     dsloss1 = nn.BCELoss()(l1, label)
    #     dsloss2 = nn.BCELoss()(l2, label)
    #     Dice_loss = 0.5*(dsloss1+dsloss2)

    #     label = torch.argmax(label, 1).squeeze(1)
    #     # label = torch.to_tensor(label, torch.float16)

    #     CT_loss = nn.CrossEntropyLoss()(prob, label)
    #     CD_loss = CT_loss + wdice * Dice_loss
    #     return CD_loss
    
    # @staticmethod
    # def predict(pred):
    #     prob, _, _ = pred
    #     return prob

 