import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_models.mobilesam import build_sam_vit_t
from .decoder import DTMS, UpConvBlock

class SAM_Mamba(nn.Module):
    def __init__(self, img_size) -> None:
        super().__init__()
        self.sam = build_sam_vit_t(img_size)

        self.bf4 = DTMS(320, 64)
        self.bf2 = DTMS(160, 64)

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

        f4 = self.bf4(p4, b4)
        f4 = self.up1(f4)
        
        f2 = self.bf2(p2, b2)
        f2 = torch.cat([f4, f2], 1)
        f2 = self.up2(f2)

        f2 = self.cls(f2)
        return f2
