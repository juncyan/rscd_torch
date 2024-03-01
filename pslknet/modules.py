import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ConvBnReLU, DepthWiseConv2D

from .blocks import *

class BFELKB(nn.Module):
    ##bi-temporal feature extraction based large kernel block
    def __init__(self, in_channels, out_channels, kernels = 7, stride=2):
        super().__init__()
        self.fe = LKFE(in_channels, kernels)
        self.ce = LKCE(in_channels, out_channels, kernels, stride)
        
    def forward(self, x):
        y = self.fe(x)
        y = self.ce(y)
        return y

class STAF(nn.Module):
    #Spatial and Temporal Adaptive Fusion Module
    def __init__(self, in_channels=3, out_channels=64, kernels=7):
        super().__init__()

        self.conv1 = DepthWiseConv2D(in_channels, kernels, 1)
        self.conv2 = DepthWiseConv2D(in_channels, kernels, 1)
        self.cbr1 = ConvBnReLU(2*in_channels, out_channels, 3, stride=2, padding=1)
        self.dws = ConvBnReLU(out_channels, out_channels, 3, 1, 1)
        self.cbr2 = ConvBnReLU(out_channels, out_channels, 3, 1, 1)
        
        self.tdcbrs2 = ConvBnReLU(2*in_channels, out_channels, 1, stride=2)
        self.tdc11 = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.tddsc = nn.Conv2d(out_channels, out_channels, 7, 1, 3, groups=out_channels)
        self.tdcbr2 = ConvBnReLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        ym = torch.concat([y1, y2], 1)
        
        ym = self.cbr1(ym)
        y = self.dws(ym)
        y = self.cbr2(y)

        Td = torch.concat([x1, x2],1)
        td = self.tdcbrs2(Td)
        td1 = self.tdc11(td)
        td2 = self.tddsc(td)
        tc = td1 + td2
        td = self.tdcbr2(tc)

        res = y + td
        return res

class PSAA(nn.Module):
    #pseudo siamese bi-temporal assimilating assistant module
    def __init__(self, mid_channels=[64, 128, 256, 512]):
        super().__init__()
        self.branch1 = FEBranch(3, mid_channels)
        self.branch2 = FEBranch(3, mid_channels)

    def forward(self, x1, x2):
        # x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i + j
            res.append(z)
        return res

class SBFA(nn.Module):
    #siamese bi-temporal feature assimilating module
    def __init__(self, mid_channels=[64, 128, 256, 512]):
        super().__init__()
        self.branch1 = FEBranch(3, mid_channels)
        # self.branch2 = FEBranch(3, mid_channels)

    def forward(self, x1, x2):
        # x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch1(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i + j
            res.append(z)
        return res

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__() 
        mid_c = out_channels * 2
        self.cbr1 = ConvBnReLU(in_channels, out_channels, 3, 1, 1)
        self.cbr2 = ConvBnReLU(out_channels, mid_c, 1)
        self.cbr3 = ConvBnReLU(mid_c, out_channels, 3, 1, 1)

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1,x2.shape[2:],mode='bilinear')
        x = torch.concat([x1, x2], axis=1)
        y = self.cbr1(x) 
        y = self.cbr2(y)
        res = self.cbr3(y)
        return res

class MF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cbr = ConvBnReLU(in_channels, out_channels, 3, 1, 1)
    
    def forward(self, x1, x2):
        x = torch.concat([x1, x2], 1)
        y = self.cbr(x)
        return y