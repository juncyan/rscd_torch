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

from cd_models.ultralight_unet import act_layer, _init_weights

from .utils import ConvBNAct
from .replk import SS2D_v3, DilatedReparamBlock


class ChannelSSM(nn.Module):
    def __init__(self, dim, out_channel=None):
        super().__init__()
        out_channel = out_channel if out_channel else dim
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.cbr = ConvBNAct(dim, out_channel, bias=False)
        self.ssm = SS2D_v3(dim, out_channel)

    def forward(self, x):
        y1 = self.avg(x)
        y1 = self.cbr(y1)
        y2 = self.ssm(x)
        y = x * (y1 + y2)
        return y


class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.replk = nn.Sequential(DilatedReparamBlock(dim, 13),
                                   SS2D_v3(dim, dim),
                                   act_layer('relu6'))

        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=False)
        self.oper_q = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.oper_k = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = ChannelSSM(dim, dim)#ConvBNAct(dim, dim, 3, 1, padding=1)
        # self.proj_drop = nn.Dropout(0.)

    def forward(self, x):
        y = self.replk(x)
        q, k, v = self.qkv(y).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        # out = self.proj_drop(out)
        return out


class RepLKSSMLayer(nn.Module):
    def __init__(self, in_channels, kernel_size=13, activation='relu6'):
        super().__init__()
        self.dwconvs = nn.Sequential(
                DilatedReparamBlock(in_channels, kernel_size),
                SS2D_v3(in_channels, in_channels),
                act_layer(activation, inplace=True))
    
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        y = self.dwconvs(x)
        return y


class RepLKSSMBlock(nn.Module):
    def __init__(self, in_channels, kernel_sizes=[7, 13] , activation='relu6'):
        super().__init__()
        self.dwconvs = nn.ModuleList([
            nn.Sequential(RepLKSSMLayer(in_channels, kernel_size, activation))
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


class CrossDimensionalGroupedAggregation(nn.Module):
    def __init__(self,F_g, F_l, F_int, activation='relu'):
        super().__init__()
        
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1,stride=1)

        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,)


        self.psi = nn.Sequential(
            act_layer('relu'),
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.rlk = nn.Sequential(RepLKSSMLayer(F_int, 13, activation))

        # self.ssm = SS2D_v3(F_int, F_int)

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        y = (g1 + x1).contiguous()

        y1 = self.rlk(y)
        y2 = self.psi(y)
        y2 = y2 * x
        y = y2 + y1
        return y