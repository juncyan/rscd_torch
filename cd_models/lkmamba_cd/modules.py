import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import os
from functools import partial
import math

import timm
from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

from cd_models.ultralight_unet import _init_weights

from .lkmamba import SS2D_v3, LKSSMBlock

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, groups=1, padding=0, bias=True, act='relu',channel_first=True):
        super().__init__()
        self.channel_first = channel_first
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        if act == "silu":
            self.act = nn.SiLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        elif act == 'softmax':
            self.act = nn.Softmax()
        elif act == "sigmoid":
            self.act == nn.Sigmoid()
    
    def forward(self, input):
        if self.channel_first:
            x = input
        else:
            x = input.permute(0, 3, 1, 2)
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        if self.channel_first:
            return y
        y = y.permute(0, 2, 3, 1)
        return y

class LKFA(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[7, 13] , activation='relu6'):
        super().__init__()
        self.dwconvs = nn.ModuleList([
            LKSSMBlock(in_channels, kernel_size)
            for kernel_size in kernel_sizes])
            
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        y = x.contiguous()
        for dwconv in self.dwconvs:
            y1 = dwconv(y)
            y = (y + y1).contiguous()
      
        return y


class BCIF(nn.Module):
    def __init__(self, in_ch, out_ch, kernels_size=13, channel_first=True):
        super().__init__()
        self.channel_first = channel_first
        dim1 = 2*in_ch
        self.ssm1 = SS2D_v3(dim1, out_ch)
        self.ssm2 = SS2D_v3(in_ch, out_ch)
        
        self.conv = ConvBNAct(2*in_ch, out_ch, 1)
        self.cbr1 = ConvBNAct(3*out_ch, out_ch, 1)
        self.cbr2 = LKSSMBlock(out_ch, kernels_size)
        self.cbr3 = ConvBNAct(out_ch, out_ch, 3, padding=1)

    def forward(self, x1, x2):
        _device = x1.device
        B, C, H, W = x1.size()
        tp = torch.cat([x1, x2], dim=1)
        tp = self.conv(tp)

        t1 = torch.empty(B, 2*C, H, W).cuda(_device)
        # t1 = torch.cat([x1, x2], dim=-1)
        t1[:, 0::2, :, :] = x1
        t1[:, 1::2, :, :] = x2
        
        t1 = self.ssm1(t1)  

        t2 = torch.empty(B, C, H, 2*W).cuda(_device)
        # x1 = self.cbr(x1)
        t2[:, :, :, 0::2] = x1
        t2[:, :, :, 1::2] = x2
        # # t2 = t2.permute(0,2,3,1)
        # t2 = torch.cat([x1, x2], 2)
        t2 = self.ssm2(t2)
        
        t = torch.cat([t1, t2[:, :, :, :W], t2[:, :, :, W:]], dim=1).contiguous()
        # t = self.ssm3(t)
        # t = t.permute(0, 3, 1, 2)
        t = self.cbr1(t)
        t = self.cbr2(t)
        t = self.cbr3(t)
        
        t = t + tp
        
        return t