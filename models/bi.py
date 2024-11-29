# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn, relu
from torch.nn import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F

class Refine(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Refine, self).__init__()
        self.inchannel = inchannel
        self.outchannel = outchannel
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inchannel, self.inchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.inchannel + self.outchannel, self.outchannel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(self.outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, x2.size()[2:], mode='bilinear', align_corners=True)  #512
        x1 = self.conv1(x1)
        x_f = torch.cat([x1, x2], dim=1)
        x_f = self.conv2(x_f)
        return x_f

class ChangeInformationExtractionModule(nn.Module):
    def __init__(self, in_d, out_d):
        super(ChangeInformationExtractionModule, self).__init__()

        self.in_d = in_d
        self.out_d = out_d

        self.conv_dr = nn.Sequential(
            nn.Conv2d(64, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )
        self.pools_sizes = [2, 4, 8]
        self.refine1 = Refine(512, 256)
        self.refine2 = Refine(256, 128)
        self.refine3 = Refine(128, 64)
        self.conv_pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[0], stride=self.pools_sizes[0]),  # 0 0
            nn.Conv2d(self.in_d, 128, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[1], stride=self.pools_sizes[1]),  # 1 1
            nn.Conv2d(self.in_d, 256, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.conv_pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.pools_sizes[2], stride=self.pools_sizes[2]),  # 2 2
            nn.Conv2d(self.in_d, 512, kernel_size=3, stride=1, padding=1, bias=False)
        )


    def forward(self, d5, d4, d3, d2):

        # refine
        r1 = self.refine1(d5, d4)
        r2 = self.refine2(r1, d3)
        x = self.refine3(r2, d2)

        x = self.conv_dr(x)

        # pooling
        p2 = x
        p3 = self.conv_pool1(x)
        p4 = self.conv_pool2(x)
        p5 = self.conv_pool3(x)

        return p5, p4, p3, p2


# 离散化输出
class PriFUwithBN(nn.Module):
    def __init__(
            self,
            num_features,
            affine=False,
            threshold =1e-1          
    ):
        super(PriFUwithBN, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=affine)
        self.feat_filter=FeatFilter(num_features, alpha=threshold)

    def forward(self, x):
        x = self.bn(x)
        out = self.feat_filter(x)
        return out


class FeatFilter(nn.Module):
    def __init__(
            self,
            num_features,
            alpha =2e-1          
    ):
        super(FeatFilter, self).__init__()
        self.alpha = alpha
        self.num_features = num_features
        self.weight = Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = Parameter(torch.ones(1, num_features, 1, 1))
        self.mask_weight = None
        nn.init.constant_(self.weight, 0.1)
        nn.init.constant_(self.bias, 0)


    def forward(self, x):
        x, weight = RedistributeGrad.apply(x, self.weight)
        threshold = self.alpha * torch.max(weight,dim=1,keepdim=True).values
        mask = 0.5*(1+torch.sign(self.weight - threshold))
        self.mask_weight = mask * self.weight
        out = self.mask_weight * x + self.bias
        return out

class RedistributeGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):

        ctx.save_for_backward(weight)
        return x, weight

    @staticmethod
    def backward(ctx, grad_x, grad_weight):

        weight, = ctx.saved_tensors
        # set the beta
        return grad_x, grad_weight + weight * 1e-3