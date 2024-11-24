import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_models.mobilesam import build_sam_vit_t
# from cd_models.vmamba.mamba_backbone import Backbone_VSSM
from cd_models.unireplknet import  unireplknet_s, unireplknet_b

from .decoder import DTMS, UpConvBlock, Decoder, DTMS_v1, Decoder_v1
from .replk import SS2D_v3
from .ram import ChannelSSM
from .mkdc import CrossDimensionalGroupedAggregation, RepLKSSMBlock

# RepLK Convolutional Additive Mamba for Land Cover Fain-graind Understanding


class RepLKSSM_CD_v2(nn.Module):
    def __init__(self, num_cls=2) -> None:
        super().__init__()
        self.encoder = unireplknet_s()
        self.encoder.eval()
        self.encoder.reparameterize_unireplknet()

        self.bf4 = DTMS_v1(768, 64)
        self.bf2 = DTMS_v1(192, 64)

        self.up1 = RepLKSSMBlock(64)
        self.up2 = RepLKSSMBlock(64)

        self.df = CrossDimensionalGroupedAggregation(64,64,64)

        self.cls = nn.Sequential(ChannelSSM(64,64),
                                 nn.Conv2d(64, 2, 7, padding=3))

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        f1_list = self.encoder(x1)
        f2_list = self.encoder(x2)
        
        p4, p2 = f1_list[3], f1_list[1]
        b4, b2 = f2_list[3], f2_list[1]
        
        f4 = self.bf4(p4, b4).contiguous()
        f2 = self.bf2(p2, b2).contiguous()
        
        f4 = self.up1(f4)
        f4 = F.interpolate(f4, f2.shape[-2:], mode='bilinear')
        f = self.df(f4, f2)
        f = self.up2(f)
        f = F.interpolate(f, scale_factor=8, mode='bilinear')
        
        f = self.cls(f)
        return f


class RepLKSSM_CD_v1(nn.Module):
    def __init__(self, num_cls=2) -> None:
        super().__init__()
        self.encoder = unireplknet_s()
        self.encoder.eval()
        self.encoder.reparameterize_unireplknet()

        self.bf4 = DTMS_v1(768, 64)
        self.bf2 = DTMS_v1(192, 64)

        self.decoder = Decoder_v1()

        self.cls = nn.Conv2d(64,2, 7, padding=3)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        f1_list = self.encoder(x1)
        f2_list = self.encoder(x2)
        
        p4, p2 = f1_list[3], f1_list[1]
        b4, b2 = f2_list[3], f2_list[1]
        
        f4 = self.bf4(p4, b4).contiguous()
        f2 = self.bf2(p2, b2).contiguous()
        
        f = self.decoder(f2, f4)
        
        f = self.cls(f)
        return f
    


class RepLKSSM_CD(nn.Module):
    def __init__(self, num_cls=2) -> None:
        super().__init__()
        self.encoder = unireplknet_s()
        self.encoder.eval()
        self.encoder.reparameterize_unireplknet()

        self.bf4 = DTMS(768, 64)
        self.bf2 = DTMS(192, 64)

        self.decoder = Decoder()

        self.cls = nn.Conv2d(64,2, 7, padding=3)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        f1_list = self.encoder(x1)
        f2_list = self.encoder(x2)
        
        p4, p2 = f1_list[3], f1_list[1]
        b4, b2 = f2_list[3], f2_list[1]
        
        f4 = self.bf4(p4, b4)
        f2 = self.bf2(p2, b2)
        
        f = self.decoder(f2, f4)
        
        f = self.cls(f)
        return f
    

