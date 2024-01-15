import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_models.utils import ConvBn, ConvBnReLU
from core.models.danet import _PositionAttentionModule
from .modules import CDFSF, BFE
from .blocks import BMF

from cd_models.resnet import resbone101, resbone50, resbone34

class MSFGNet(nn.Module):
    #multi-scale feature gather network
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.bmf = BMF(3)
        self.encode = BFE()

        self.pam = _PositionAttentionModule(512)

        self.up1 = CDFSF(512, 512)
        self.up2 = CDFSF(512, 256)
        self.up3 = CDFSF(256, 128)
        self.up4 = CDFSF(128, 64)
        
        self.classier = nn.Sequential(ConvBn(64, num_classes, 7, 1,3), nn.Sigmoid())

    def forward(self, x1, x2):
        # x1 ,x2 = x[:, :3, :, :], x[:, 3:, :, :]
        f1 = self.bmf(x1,x2)
        feature1 = self.encode(f1)
        f2, f3, f4 = feature1
        
        f5 = self.pam(f4)

        y = self.up1(f5,f4)
        y = self.up2(y, f3)
        y = self.up3(y, f2)
        y = self.up4(y, f1)
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        return self.classier(y)


class MSFGNet_Res101(nn.Module):
    #multi-scale feature gather network
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.bmf = BMF(3)
        self.encode = resbone101()

        self.pam = _PositionAttentionModule(4*512)

        self.up1 = CDFSF(4*512, 4*512)
        self.up2 = CDFSF(4*512, 4*256)
        self.up3 = CDFSF(4*256, 4*128)
        self.up4 = CDFSF(4*128, 64)
        
        self.classier = nn.Sequential(ConvBn(64, num_classes, 3, 1, 1), nn.Sigmoid())

    def forward(self, x1, x2):
        # x = torch.cat([x1,x2], 1)
        f1 = self.bmf(x1, x2)
        feature1 = self.encode(f1)
        f2, f3, f4 = feature1
        
        f5 = self.pam(f4)

        y = self.up1(f5,f4)
        y = self.up2(y, f3)
        y = self.up3(y, f2)
        y = self.up4(y, f1)
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        return self.classier(y)

class Up(nn.Module):
    def __init__(self, in_c1, in_c2):
        super().__init__()
        in_channels = in_c1 + in_c2
            #self.up = nn.functional.interpolate(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = ConvBnReLU(in_channels, in_c2 , 1)
        # self.conv2 = ConvBnReLU(in_c2, in_c2, 3, padding=1)
        
    def forward(self, x1, x2):
        if x1.shape != x2.shape:
            x1 = F.interpolate(x1, x2.shape[-2:], mode='bilinear')

        x = torch.concat([x1, x2], 1)
        
        return self.conv1(x)

class MSFGNet_noCDFSFl(nn.Module):
    #multi-scale feature gather network
    def __init__(self,in_channels=6, num_classes=2):
        super().__init__()
        self.bmf = BMF(3)
        self.encode = BFE()

        self.pam = _PositionAttentionModule(512)

        self.up1 = Up(512, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        
        self.classier = nn.Sequential(ConvBn(64, num_classes, 3, 1,1), nn.Sigmoid())

    def forward(self, x1, x2):
        # x = torch.cat([x1,x2], 1)
        f1 = self.bmf(x1, x2)
        feature1 = self.encode(f1)
        f2, f3, f4 = feature1
        
        f5 = self.pam(f4)

        y = self.up1(f5,f4)
        y = self.up2(y, f3)
        y = self.up3(y, f2)
        y = self.up4(y, f1)
        y = F.interpolate(y, scale_factor=2, mode="bilinear")
        return self.classier(y)
