import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_models.mobilesam import build_sam_vit_t
from cd_models.vmamba.mamba_backbone import Backbone_VSSM
from cd_models.unireplknet import  unireplknet_s, unireplknet_b

from .decoder import DTMS, UpConvBlock
from .replk import SS2Dv_Lark

class SAM_Mamba(nn.Module):
    def __init__(self, img_size=512) -> None:
        super().__init__()
        self.sam = Backbone_VSSM(out_indices=(0, 1, 2, 3)) #build_sam_vit_t(img_size)

        self.bf4 = DTMS(1024, 64)
        self.bf2 = DTMS(256, 64)

        self.up1 = UpConvBlock(64,64,4)
        self.up2 = UpConvBlock(128,64,8)

        self.cls = nn.Conv2d(64,2, 7, padding=3)

        for param in self.sam.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        f1_list = self.sam(x1)
        f2_list = self.sam(x2)
        
        p4, p2 = f1_list[3], f1_list[1]
        b4, b2 = f2_list[3], f2_list[1]

        p4 = p4.permute(0,2,3,1)
        b4 = b4.permute(0,2,3,1)
        p2 = p2.permute(0,2,3,1)
        b2 = b2.permute(0,2,3,1)

        f4 = self.bf4(p4, b4)
        f4 = self.up1(f4)
        f2 = self.bf2(p2, b2)
        
        f2 = torch.cat([f4, f2], 1)
        f2 = self.up2(f2)

        f2 = self.cls(f2)
        return f2


class LargeMamba(nn.Module):
    def __init__(self, img_size=512) -> None:
        super().__init__()
        self.encoder = unireplknet_s()
        self.encoder.eval()
        self.encoder.reparameterize_unireplknet()

        self.bf4 = DTMS(768, 64)
        self.bf2 = DTMS(192, 64)

        self.up1 = UpConvBlock(64,64,4)
        self.up2 = UpConvBlock(128,64,8)

        self.cls = nn.Conv2d(64,2, 7, padding=3)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        f1_list = self.encoder(x1)
        f2_list = self.encoder(x2)
        
        p4, p2 = f1_list[3], f1_list[1]
        b4, b2 = f2_list[3], f2_list[1]
        # print(p4.shape, p2.shape)
        p4 = p4.permute(0,2,3,1)
        b4 = b4.permute(0,2,3,1)
        p2 = p2.permute(0,2,3,1)
        b2 = b2.permute(0,2,3,1)
        f4 = self.bf4(p4, b4)
        f4 = self.up1(f4)
        
        f2 = self.bf2(p2, b2)
        f2 = torch.cat([f4, f2], 1)
        f2 = self.up2(f2)

        f2 = self.cls(f2)
        return f2