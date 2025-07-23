
import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, vit_b_16, swin_v2_b
from cd_models.lccdmamba.backbone import Backbone_VSSM
from einops import rearrange, repeat


class CIENet_VMB(nn.Module):
    def __init__(self, img_size, num_seg=7, **kwargs):
        super().__init__()
        self.backbone = Backbone_VSSM((0,3))
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.img_size = [img_size, img_size]
        self.clim = CrossLevelInformationMerge(self.backbone.dims[-1], self.backbone.dims[0], 64)
        
        self.bdia = BitemporalDifferenceInformationAggregation(64, 64)
        self.cls = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                 nn.BatchNorm2d(64),
                                 nn.GELU(), 
                                 nn.Conv2d(64, 1, 3, 1, 1))

        self.dmfe = DualMaskFeatureEnhance(64)
        self.scls1 = nn.Sequential(nn.Conv2d(64,64,3,1,1),
                                   nn.BatchNorm2d(64),
                                   nn.GELU(), 
                                   nn.Conv2d(64, num_seg, 3, 1, 1))
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = torch.split(x1, 2, dim=1)
            x1 = x[0]
            x2 = x[1]
    
        f1, f2 = self.encoder(x1)
        p1, p2 = self.encoder(x2)
        
        f2 = self.clim(f2, f1)
        p2 = self.clim(p2, p1)
 
        y = self.bdia(f2, p2)
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)

        s1 = self.dmfe(f2)
        s2 = self.dmfe(p2)
        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
    
    def encoder(self, x):   
        y = self.backbone(x)
        return y


class DualMaskFeatureEnhance(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, 2*in_c, 1), LayerNorm2d(2*in_c))

        self.gconv1 = nn.Sequential(GhostConv2D(in_c, in_c, 3, 1, 1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU())
        self.dwc1 = nn.Sequential(nn.Conv2d(in_c, in_c, 3, padding=1),
                                  nn.BatchNorm2d(in_c),
                                  nn.ReLU())

        self.mlp2 = nn.Sequential(nn.Conv2d(in_c, in_c, 1),
                                  nn.GELU(),
                                  nn.Conv2d(in_c, in_c, 1))

        self.conv = nn.Sequential(nn.Conv2d(2*in_c, in_c, 1),
                                  nn.BatchNorm2d(in_c),
                                  nn.GELU())
    
    def forward(self, x):
        y = self.proj(x)
        
        x1, x2 = torch.chunk(y, 2, dim=1)
        y1 = self.gconv1(x1)
        y1 = self.dwc1(y1)

        y2 = F.adaptive_avg_pool2d(x2, 1)
        y2 = self.mlp2(y2)
        y2 = x2 * y2
        z = torch.concat([y1, y2], dim=1)
        z = self.conv(z)
        return z


class CrossLevelInformationMerge(nn.Module):   
    def __init__(self, in_ch1, in_ch2, out_chs):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch1, out_chs, 3, padding=1),
                                  nn.BatchNorm2d(out_chs),
                                  nn.ReLU())
        self.proj2 = nn.Sequential(nn.Conv2d(in_ch2, out_chs, 3, padding=1),
                                  nn.BatchNorm2d(out_chs),
                                  nn.ReLU())
        self.sa = nn.Sequential(
                    nn.Conv2d(out_chs, in_ch1, kernel_size=1),
                    nn.BatchNorm2d(in_ch1),
                    nn.Conv2d(in_ch1, out_chs, kernel_size=1),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Sigmoid(),
                    )
        self.lamda = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        
        self.conv3d = nn.Sequential(nn.Conv3d(2, 2, 3, padding=1), nn.BatchNorm3d(2), nn.ReLU(), 
                                    nn.Conv3d(2, 1, 3, padding=1))
        
        self.dwc0 = nn.Sequential(nn.Conv2d(out_chs, out_chs, 1),
                                  nn.BatchNorm2d(out_chs))
        self.dwc1 = nn.Sequential(nn.Conv2d(out_chs, out_chs, 5, padding=2, groups=out_chs),
                                  nn.BatchNorm2d(out_chs))
        self.dwc2 = nn.Sequential(nn.Conv2d(out_chs, out_chs, 7, padding=3, groups=out_chs),
                                  nn.BatchNorm2d(out_chs))
       
        self.cbr = nn.Sequential(nn.Conv2d(out_chs, out_chs, 3, padding=1),
                                  nn.BatchNorm2d(out_chs),
                                  nn.ReLU())
    
    def forward(self, x, y):
        x = self.proj(x)
        sax = self.sa(x)

        x1 = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=True)
        x2 = self.dwc0(x1) + self.dwc1(x1) + self.dwc2(x1)

        y = self.proj2(y)
        x3 = torch.stack([x1, y], dim=1)
        x3 = self.conv3d(x3)

        x3 = x3.squeeze(1)
        f = x2 + self.lamda * x3
        f = f * sax
        f = self.cbr(f)
        return f


class BitemporalDifferenceInformationAggregation(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # self.zip = nn.Conv2d(256, in_c, 1)
        self.conv1 = nn.Conv2d(in_c, in_c, 3, padding=1)
        self.dwc = nn.Sequential(nn.Conv2d(in_c, in_c, 5, padding=2, groups=in_c),
                                  nn.BatchNorm2d(in_c))
        self.conv2 = nn.Conv2d(in_c, in_c, 3, padding=1)
        # self.cbr1 = layers.ConvBNReLU(in_c, in_c, 3)

        self.cbr = nn.Sequential(nn.Conv2d(in_c, out_c, 3, padding=1),
                                  nn.BatchNorm2d(out_c),
                                  nn.ReLU())
    
    def forward(self, x, y):
        # x = self.zip(x)
        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x1 = self.conv1(x)
        y1 = self.conv1(y)
        f = x1 - y1
        f = self.conv2(f)
        x2 = self.dwc(x1)
        y2 = self.dwc(y1)
        # x2 = x2 + f
        # x2 = self.cbr1(x2)
        # y2 = y2 + f
        # y2 = self.cbr1(y2)
        # f = x2+y2
        f = (x2 + y2) * f
        f = self.cbr(f)
        return f

class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(*normalized_shape, 1, 1))
            self.bias = nn.Parameter(torch.zeros(*normalized_shape, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        # 计算均值和方差时，只在通道维度上计算
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)
        
        # 归一化
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # 仿射变换
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias
            
        return x_normalized

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

class GhostConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        init_ch = out_channels // 2
        
        
        # 生成偏移量的卷积层
        self.prim_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU()
        )
        self.cheap_conv = nn.Sequential(
            nn.Conv2d(init_ch, init_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=init_ch),
            nn.BatchNorm2d(init_ch),
            nn.ReLU()
        )
    def forward(self, x):
        x1 = self.prim_conv(x)
        x2 = self.cheap_conv(x1)
        output = torch.concat([x1, x2], dim=1)
        
        return output