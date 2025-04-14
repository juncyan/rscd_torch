
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
from cd_models.vmamba import SS2D, Backbone_VSSM
from .sfhformer import FFN
from .utils import ConvBNAct


class Local_Feature_Gather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dim_sp = dim//2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):

        x1 = x[:, :self.dim_sp, :, :]
        x2 = x[:, self.dim_sp:, :, :]

        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.cat([cd1, cd2], dim=1)

        return x


class Global_Feature_Gather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim*2, 1),
            nn.GELU()
        )
        self.conv_fina = nn.Sequential(
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.token = SS2D(self.dim*2)

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.token(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv_fina(x+x0)

        return x

class Mixer(nn.Module):
    def __init__(self, dim):
        super(Mixer, self).__init__()
        self.dim = dim
        self.mixer_local = Local_Feature_Gather(dim=self.dim)
        self.mixer_gloal = Global_Feature_Gather(dim=self.dim)

        self.conv_init = nn.Conv2d(dim, dim * 2, 1)
        
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()
        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        

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

class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.dim = dim
        self.norm1 = nn.BatchNorm2D(dim)
        self.norm2 = nn.BatchNorm2D(dim)
        self.mixer = Mixer(dim=self.dim)
        # self.ffn = FFN(dim=self.dim)

        self.beta = self.create_parameter(shape=[1, dim, 1, 1], default_initializer=nn.initializer.Constant(value=0.0))
        # self.gamma = self.create_parameter(shape=[1, dim, 1, 1], default_initializer=nn.initializer.Constant(value=0.0))

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x * self.beta + copy

        # copy = x
        # x = self.norm2(x)
        # x = self.ffn(x)
        # x = x * self.gamma + copy

        return x

class VMamba_CD(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.extract = Backbone_VSSM()

        self.mixter1 = Mixer(768)
        self.conv1 = nn.Conv2d(768, 64, 1)

        self.mixter2 = Mixer(384)
        self.conv2 = nn.Conv2d(384, 64, 1)

        self.mixter3 = Mixer(192)
        self.conv3 = nn.Conv2d(192, 64, 1)

        self.mixter4 = Mixer(96)
        self.conv4 = nn.Conv2d(96, 64, 1)

        self.conv5 = ConvBNAct(64*4, 64, 1, act='relu')
        self.cls = nn.Conv2d(64, 2, 1)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = torch.split(x1, 3, dim=1)
            x1 = x[0]
            x2 = x[1]

        sz = x1.shape[-2:]
    
        f2, f3, f4, f5 = self.extract(x1)
        p2, p3, p4, p5 = self.extract(x2)
        # [2, 96, 64, 64] [2, 192, 32, 32] [2, 384, 16, 16] [2, 768, 8, 8]
        y5 = f5 + p5
        y5 = self.mixter1(y5)
        y5 = self.conv1(y5)
        y5 = F.interpolate(y5, size=sz, mode='bilinear', align_corners=True)

        y4 = self.mixter2(f4+p4)
        y4 = self.conv2(y4)
        y4 = F.interpolate(y5, size=sz, mode='bilinear', align_corners=True)

        y3 = self.mixter3(f3+p3)
        y3 = self.conv3(y3)
        y3 = F.interpolate(y3, size=sz, mode='bilinear', align_corners=True)

        y2 = self.mixter4(f2+p2)
        y2 = self.conv4(y2)
        y2 = F.interpolate(y2, size=sz, mode='bilinear', align_corners=True)

        y = torch.concat([y2, y3, y4, y5], dim=1)
        y = self.conv5(y)
        y = self.cls(y)
        return y
    
    # def train(train_loader, network, criterion, optimizer):
    #     losses = np.array([0.])
    #     for batch in train_loader:
    #         source_img = batch['source'].cuda()
    #         target_img = batch['target'].cuda()

    #         pred_img = network(source_img)
    #         label_img = target_img
    #         l3 = criterion(pred_img, label_img)
    #         loss_content = l3

    #         label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
    #         label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

    #         pred_fft3 = torch.fft.fft2(pred_img, dim=(-2, -1))
    #         pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

    #         f3 = criterion(pred_fft3, label_fft3)
    #         loss_fft = f3


    #         loss = loss_content + 0.1 * loss_fft
    #         losses.update(loss.item())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     return losses.avg