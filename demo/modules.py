import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum

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

class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, down_ratio=2):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 1, stride=1, padding=0)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, stride=down_ratio, padding=1, act='silu')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Local_Feature_Gather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_sp = dim // 2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.concat([cd1, cd2], dim=1)
        return x

class Global_Feature_Gather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU())
        
        self.token = ConvBNAct(self.dim * 2, self.dim * 2, 3, act='silu', padding=1)
         # layers.ConvBNReLU(self.dim * 2, self.dim * 2, 3)
        self.conv_fina = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.GELU()) 

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = self.token(x)
        x = self.conv_fina(x + x0)
        return x

class MultiScaleFeatureGather(nn.Module):
    def __init__(self, dim):
        super(MultiScaleFeatureGather, self).__init__()
        self.dim = dim
        self.conv_init = ConvBNAct(dim, dim * 2, 1, stride=1, padding=0, act='gelu')
        # layers.ConvBNReLU(dim, dim * 2, 1)

        self.mixer_local = Local_Feature_Gather(dim=self.dim)
        self.mixer_gloal = Global_Feature_Gather(dim=self.dim)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid())

        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_init(x)
        x = torch.split(x, self.dim, dim=1)
        
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.concat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)

        return x

class Decoder(nn.Module):
    def __init__(self, img_size, channels = [64,256]):
        super(Decoder, self).__init__()
        self.img_size = [img_size, img_size]
        # self.mixter11 = GlobalTokenAttention(channels[-1])
        self.mixter1 = MultiScaleFeatureGather(channels[-1])
        # self.conv1 = layers.ConvBNAct(channels[-1], channels[0], 3, act='gelu') #nn.Conv2d(channels[-1], channels[0], 1)

        self.mixter2 = MultiScaleFeatureGather(channels[0])  
        # self.conv2 = layers.ConvBNAct(channels[0], 64, 3, act='gelu') #nn.Conv2d(channels[0], 64, 1)

        self.cdfa = ParallChangeInformationFusion(channels[-1], channels[0], 64)
        self.conv5 = ConvBNAct(64, 64, 3, padding=1)

    def forward(self, x1, x2):
        y5 = self.mixter1(x2)

        y4 = self.mixter2(x1)
        # y4 = self.conv2(y4)
        y4 = self.cdfa(y5, y4)
        y = F.interpolate(y4, size=self.img_size, mode='bilinear', align_corners=True)

        y = self.conv5(y)
        
        return y

class SegDecoder(nn.Module):
    def __init__(self, img_size, channels = [128,160,320,256]):
        super(SegDecoder, self).__init__()
        self.img_size = [img_size, img_size]
        self.mixter1 = MultiScaleFeatureGather(channels[3])
        self.conv1 = nn.Conv2d(channels[3], channels[2], 1)

        self.mixter2 = MultiScaleFeatureGather(channels[2])
        self.conv2 = nn.Conv2d(channels[2], channels[1], 1)

        self.mixter3 = MultiScaleFeatureGather(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[0], 1)

        self.mixter4 = MultiScaleFeatureGather(channels[0])
        self.conv4 = nn.Conv2d(channels[0], 64, 1)

        self.conv5 = ConvBNAct(64, 64, 3, padding=1)

    def forward(self, f2, f3, f4, f5):
        y5 = self.mixter1(f5)
        y5 = self.conv1(y5)
        # y5 = F.interpolate(y5, size=self.img_size, mode='bilinear', align_corners=True)
        f4 = f4 + y5
        y4 = self.mixter2(f4)
        y4 = self.conv2(y4)
        # y4 = F.interpolate(y5, size=self.img_size, mode='bilinear', align_corners=True)
        f3 = f3 + y4
        y3 = self.mixter3(f3)
        y3 = self.conv3(y3)
        y3 = F.interpolate(y3, scale_factor=2, mode='bilinear', align_corners=True)
        f2 = f2 + y3
        y2 = self.mixter4(f2)
        y2 = self.conv4(y2)
        y2 = F.interpolate(y2, size=self.img_size, mode='bilinear', align_corners=True)

        # y = torch.concat([y2, y3, y4, y5], dim=1)
        y = self.conv5(y2)
        
        return y
    

class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
       
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.oper_q = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.oper_k = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Sequential(nn.Conv2d(dim, dim, 3,1, padding=1, groups=dim),
                                  nn.BatchNorm2d(dim))
        self.proj_drop = nn.Dropout(0.)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out

class ParallChangeInformationFusion(nn.Module):
    def __init__(self,in_chs1, in_chs2, out_chs):
        super().__init__()
        
        self.W_g = nn.Conv2d(in_chs1, out_chs, kernel_size=1,stride=1)
        self.W_x = nn.Conv2d(in_chs2, out_chs, kernel_size=1,stride=1)

        self.token = ConvBNAct(out_chs, out_chs, 3, act='silu', padding=1)

        self.psi = nn.Sequential(
            nn.ReLU(),
            ConvBNAct(out_chs, 1, 1, act='sigmoid'),
        )
        self.cbr = ConvBNAct(out_chs, out_chs, 3, act='silu', padding=1)
        

    def forward(self, g, x):
        sz = x.shape[-2:]
        g1 = self.W_g(g)
        g1 = F.interpolate(g1, size=sz, mode='bilinear', align_corners=False)
        x1 = self.W_x(x)

        y = g1 + x1 
        y1 = self.token(y)
        y2 = self.psi(y)
        y2 = y2 * x1
        y2 = y2 + y1
        y2 = self.cbr(y2)
        
        return y2

class CoarseInteractiveFeaturesExtraction(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(2*dim, dim, 1)
        self.token = ConvBNAct(dim, dim, 3, act='silu', padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.ffn = FFN(dim)
    
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        device = x1.device
        x = torch.empty([B, 2*C, H, W], dtype=x1.dtype).cuda(device)
        x[:, 0::2, ...] = x1
        x[:, 1::2, ...] = x2

        x = self.conv1(x)
        x = self.token(x)
        x = self.conv2(x)

        x = x + x1 + x2
        x = self.ffn(x)

        return x


class GlobalAttention(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias_attr=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b h w c')#.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose([0,1,3,2])) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x


class FFN(nn.Module):
    def __init__(self, dim):
        super(FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.conv_init = nn.Conv2d(dim, 2*dim, 1)

        self.conv1_1 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
                        groups=self.dim_sp)
        self.conv1_2 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2,
                        groups=self.dim_sp)
        
        self.conv1_3 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3,
                        groups=self.dim_sp)

        # self.conv1_1 = nn.Sequential(
        #     nn.Conv2D(self.dim_sp, self.dim_sp, kernel_size=3, padding=1,
        #               groups=self.dim_sp),
        # )
        # self.conv1_2 = nn.Sequential(
        #     nn.Conv2D(self.dim_sp, self.dim_sp, kernel_size=3, padding=4,
        #               groups=self.dim_sp, dilation=4),
        # )
        # self.conv1_3 = nn.Sequential(
        #     nn.Conv2D(self.dim_sp, self.dim_sp, kernel_size=3, padding=7,
        #               groups=self.dim_sp, dilation=7),
        # )

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(self.dim_sp, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = list(torch.split(x, self.dim_sp, dim=1)) 
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])
        # y = paddle.concat(x, axis=1)
        # y = x[0] + x[1]
        y = x[0] + x[1] + x[2] + x[3]
        y = self.gelu(y)
        y = self.conv_fina(y)

        return y