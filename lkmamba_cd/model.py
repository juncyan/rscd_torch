import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_models.unireplknet import  unireplknet_s
from .modules import LKFA, BCIF

# RepLK Convolutional Additive Mamba for Land Cover Fain-graind Understanding
class LKMamba_CD(nn.Module):
    def __init__(self, num_cls=2) -> None:
        super().__init__()
        self.encoder = unireplknet_s()
        self.encoder.eval()
        self.encoder.reparameterize_unireplknet()

        self.bf4 = BCIF(768, 64)
        self.bf2 = BCIF(192, 64)

        self.up1 = LKFA(64)
        # self.conv1 = nn.Sequential(nn.Conv2d(64,64,1), nn.BatchNorm2d(64), nn.ReLU(), nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU())
        self.up2 = LKFA(64)

        self.conv2 = nn.Sequential(nn.Conv2d(64,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU())

        self.cls = nn.Conv2d(64, 2, 7, padding=3)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2=None):
        if x2 == None:
            x2 = x1[:, 3:, :, :]
            x1 = x1[:, :3, :, :]
        f1_list = self.encoder(x1)
        f2_list = self.encoder(x2)
        
        p4, p2 = f1_list[3], f1_list[1]
        b4, b2 = f2_list[3], f2_list[1]
        
        f4 = self.bf4(p4, b4).contiguous()
        f2 = self.bf2(p2, b2).contiguous()
        
        f4 = self.up1(f4)
        # f4 = self.conv1(f4)
        f4 = F.interpolate(f4, f2.shape[-2:], mode='bilinear')
        f = self.conv2(f4+f2)
        f = self.up2(f)
        f = F.interpolate(f, scale_factor=8, mode='bilinear')
        # f = self.conv2(f)
        f = F.sigmoid(self.cls(f))
        return f