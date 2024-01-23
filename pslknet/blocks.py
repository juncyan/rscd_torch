import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ConvBnReLU, DepthWiseConv2D

class MLKC(nn.Module):
    def __init__(self, in_channels, kernels=7):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, in_channels, 1)
        self.lkc = DepthWiseConv2D(in_channels, kernels, 1)#nn.Conv2d(in_channels, in_channels, kernels, stride=1, padding= int(kernels//2), groups=in_channels)
        self.bn = nn.BatchNorm2d(in_channels)
        self.gelu = nn.GELU()
        self.cbr = ConvBnReLU(in_channels, in_channels, 3, 1, 1)
    
    def forward(self, x):
        y1 = self.c1(x)
        yk = self.lkc(x)
        y = y1 + yk
        my = self.gelu(self.bn(y))
        res = self.cbr(my)
        return res

class LKCE(nn.Module):
    #large kernel channel expansion
    def __init__(self, in_channels, out_channels, kernels = 7, stride=1):
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, 2*in_channels, 3, stride=stride, padding=1)
        self.dwc = MLKC(2*in_channels, kernels)
        self.conv3 = ConvBnReLU(2*in_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        y = self.conv1(x)
        m = self.dwc(y)
        m = self.conv3(m)
        return m

class LKFE(nn.Module):
    #large kernel feature extraction
    def __init__(self, in_channels, kernels = 7):
        super().__init__()
        self.conv1 = ConvBnReLU(in_channels, 2 * in_channels, 3, 1, 1)
        self.dwc = MLKC(2*in_channels, kernels)
        self.conv2 = ConvBnReLU(2 * in_channels, in_channels, 3, 1, 1)
        self.ba = nn.Sequential(nn.BatchNorm2d(in_channels), nn.ReLU())

    def forward(self, x):
        m = self.conv1(x)
        m = self.dwc(m)
        # m = self.se(m)
        m = self.conv2(m)
        y = x + m
        return self.ba(y)


class LKBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=7):
        super().__init__()
        self.cbr1 = ConvBnReLU(in_channels, out_channels, 3, 2, 1)
        
        self.c11 = nn.Conv2d(out_channels, out_channels, 1)
        self.gck = DepthWiseConv2D(out_channels, kernels, 1)#nn.Conv2d(out_channels, out_channels, kernels, 1, kernels//2, groups=out_channels)
        
        self.bnr = nn.Sequential(nn.BatchNorm2d(out_channels), nn.GELU())
        self.lastcbr = ConvBnReLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        z = self.cbr1(x)
        z1 = self.c11(z)
        z2 = self.gck(z)
        y = z1 + z2
        return self.lastcbr(self.bnr(y))

class FEBranch(nn.Module):
    def __init__(self, in_channels, mid_channels: list = [16, 32, 64, 128]):
        super(FEBranch, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = 3
        for c in mid_channels:
            self.layers.append(LKBlock(in_channels, c))
            in_channels = c

    def forward(self, x):
        y = x
        res = []
        for layer in self.layers:
            y = layer(y)
            res.append(y)
        return res
