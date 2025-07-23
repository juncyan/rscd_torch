
import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from cd_models.vmamba.mamba_backbone import VSSM, Backbone_VSSM
from .modules import Decoder, CoarseInteractiveFeaturesExtraction, MultiScaleFeatureGather, ParallChangeInformationFusion

class FGFPVM_Seg(nn.Module):
    #Fine-Grained Feature Processing for Segmentation
    def __init__(self, img_size, num_cls):
        super().__init__()
        self.bk = Backbone_VSSM()
        self.bk.eval()
        self.img_size = [img_size, img_size]
        
        self.msfg1 = MultiScaleFeatureGather(96)
        self.msfg2 = MultiScaleFeatureGather(768)
        self.pcif = ParallChangeInformationFusion(768, 96, 64)
        # self.decoder = Decoder(img_size=img_size, channels=[64,256])
        self.cls = nn.Conv2d(64, num_cls, 1, 1, 0)

    
    def encoder(self, x):   
        y0, _,_, y1 = self.bk(x)
        
        return y0, y1

    def forward(self, x):

        x1, x2 = self.encoder(x)
        # x2 = self.att(x2)
        y1 = self.msfg1(x1)
        y2 = self.msfg2(x2)
        y = self.pcif(y2, y1)
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        # y = self.decoder(x1, x2)
        y = self.cls(y)
        return y

class FGFPVM_CD(FGFPVM_Seg):
    def __init__(self, img_size):
        super().__init__(img_size=img_size, num_cls=2)
        # self.img_size = img_size
        self.df1 = CoarseInteractiveFeaturesExtraction(32)
        self.df2 = CoarseInteractiveFeaturesExtraction(960)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = torch.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f1, f2 = self.encoder(x1)
        p1, p2 = self.encoder(x2)
     

        c1 = self.df1(f1, p1)
        c2 = self.df2(f2, p2)
        
        c1 = self.msfg1(c1)
        c2 = self.msfg2(c2)
        y = self.pcif(c2, c1)
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)
        # y = F.softmax(y)
        return y


class FGFPVM_SCD(FGFPVM_Seg):
    def __init__(self, img_size, num_seg=7):
        super().__init__(img_size=img_size, num_cls=1)

        self.df1 = CoarseInteractiveFeaturesExtraction(32)
        self.df2 = CoarseInteractiveFeaturesExtraction(960)

        self.pcif2 = ParallChangeInformationFusion(960, 32, 32)
        self.scls1 = nn.Conv2d(32, num_seg, 3, 1, 1)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = torch.split(x1, 2, axis=1)
            x1 = x[0]
            x2 = x[1]
    
        f1, f2 = self.encoder(x1)
        p1, p2 = self.encoder(x2)

        f1 = self.msfg1(f1)
        p1 = self.msfg1(p1)

        f2 = self.msfg2(f2)
        p2 = self.msfg2(p2)

        c1 = self.df1(f1, p1)
        c2 = self.df2(f2, p2)
        y = self.pcif(c2, c1)
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)

        s1 = self.pcif2(f2, f1)
        s2 = self.pcif2(p2, p1)
        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
    
    # @staticmethod
    # def loss(logits,labels):
    #     if len(labels.shape) == 4:
    #         labels = torch.argmax(labels, 1)
    #     if logits.shape == labels.shape:
    #         labels = torch.argmax(labels,axis=1)
    #     elif len(labels.shape) == 3:
    #         labels = labels
    #     else:
    #         assert "pred.shape not match label.shape"
    #     labels = torch.cast(labels, dtype='float32')
    #     # logits = torch.tensor.cast(logits, dtype='float64')
    #     ce_loss_1 = CrossEntropyLoss()(logits, labels)
    #     # return ce_loss_1
    #     lovasz_loss = LovaszSoftmaxLoss()(logits, labels)
    #     main_loss = ce_loss_1 + 0.75 * lovasz_loss
    #     return main_loss
    