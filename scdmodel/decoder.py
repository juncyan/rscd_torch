import torch
import torch.nn as nn
import torch.nn.functional as F
from cd_models import layers

from .utils import features_transfer
# from cd_models.mamba.ppmamba import MambaBlock
from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
from cd_models.vmamba import SS2D 
from models.cdfa import ContrastDrivenFeatureAggregation


class SemantiMambacDv0(nn.Module):
    """ spatial channel attention module"""
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        
        # conf1 = MambaConfig(4096, in_c2)
        # self.ssm1 = nn.Sequential(MambaMixer(conf1, 0), MambaMixer(conf1, in_c2-1), nn.LayerNorm(in_c2))

        # conf2 = MambaConfig(1024, in_c1)
        # self.ssm2 = nn.Sequential(MambaMixer(config=conf2, layer_idx=0), MambaMixer(conf2, in_c1-1), nn.LayerNorm(in_c1))
        self.ssm1 = SS2D(in_c2)
        self.ssm2 = SS2D(in_c1)
        self.st1conv1 = layers.ConvBNReLU(in_c1, in_c2, 1)
        self.st1conv2 = layers.ConvBNReLU(in_c2, in_c2, 3)
        self.st1conv3 = layers.ConvBNReLU(in_c2, in_c2, 1)
        
        self.sa1 = nn.Sequential(layers.ConvBN(2,1,1), nn.Sigmoid())
        
        self.st2conv2 = layers.ConvBNReLU(in_c2, out_c, 1)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 3)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 1)

    def forward(self, x1, x2):
        f = features_transfer(x2, "NWHC")
        f = self.ssm2(f)
        # f = self.fft2(f)
        f = f.permute(0, 3, 1, 2)
        
        f = self.st1conv1(f)
        max_feature1 = torch.max(f, dim=1, keepdim=True)[0]
        mean_feature1 = torch.mean(f, dim=1, keepdim=True)
        att_feature1 = torch.concat([max_feature1, mean_feature1], dim=1)
        
        y = self.sa1(att_feature1)
        f = y * f

        f1 = self.st1conv2(f)
        f1 = self.st1conv3(f1)
        f1 = F.interpolate(f1, scale_factor=8, mode='bilinear', align_corners=True) #self.up1(f)

        f2 = features_transfer(x1, "NWHC")
        f2 = self.ssm1(f2)
        # f2 = self.fft1(f2)
        f2 = f2.permute(0, 3, 1, 2)
        f2 = f1 + f2
        f2 = self.st2conv2(f2)
        f2 = F.interpolate(f2, size=self.img_size, mode='bilinear', align_corners=True)
        f2 = self.st2conv3(f2)
        f2 = self.st2conv3(f2)
        
        # f2 = f2 + f1
        return f2 

class SemantiCrossA(nn.Module):
    """ spatial channel attention module"""
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        self.st1conv1 = layers.ConvBNReLU(in_c1, in_c2, 1)
        self.st1conv2 = layers.ConvBNReLU(in_c2, in_c2, 3)
        self.st1conv3 = layers.ConvBNReLU(in_c2, in_c2, 1)

        self.fg = layers.ConvBNReLU(256,in_c2, 1)
        self.ca = ContrastDrivenFeatureAggregation(in_c2, in_c2)
        
        self.st2conv2 = layers.ConvBNReLU(in_c2, out_c, 1)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 3)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 1)

    def forward(self, x1, x2, x):
        f = features_transfer(x2, "NCWH")
        f = self.st1conv1(f)
        f1 = self.st1conv2(f)
        f1 = self.st1conv3(f1)
        
        f_ = self.fg(x)
        f2 = features_transfer(x1, "NCWH")
        f1 = F.interpolate(f1, size = f2.shape[-2:], mode='bilinear', align_corners=True)
        f_ = F.interpolate(f_, size = f2.shape[-2:], mode='bilinear', align_corners=True)

        f2 = self.ca(f2, f1, f_)
        f2 = self.st2conv2(f2)
        f2 = self.st2conv3(f2)
        f2 = self.st2conv3(f2)
        f2 = F.interpolate(f2, size=self.img_size, mode='bilinear', align_corners=True)
        
        # f2 = f2 + f1
        return f2 