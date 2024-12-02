import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from cd_models.mobilesam import build_sam_vit_t
# from cd_models.vmamba.mamba_backbone import Backbone_VSSM
from cd_models.unireplknet import  unireplknet_s, unireplknet_b
from timm.models.helpers import named_apply

from .utils import _init_weights

from .backbone import LKSSMNet
from .decoder import UpConvBlock_v1, UpConvBlock, DTMS
from .replk import SS2D_v3
from .ram import ChannelSSM, ConvBNAct
from .mkdc import CrossDimensionalGroupedAggregation, RepLKSSMBlock

# RepLK Convolutional Additive Mamba for Land Cover Fain-graind Understanding



class RepLKSSM_Seg(nn.Module):
    def __init__(self, num_cls=5) -> None:
        super().__init__()
        self.encoder = LKSSMNet() #unireplknet_s()
        # self.encoder.eval()
        # self.encoder.reparameterize_unireplknet()

        self.conv1 = nn.Sequential(nn.Conv2d(768, 128, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.lks1 = RepLKSSMBlock(128)
        self.conv11 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(192, 64, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.df = CrossDimensionalGroupedAggregation(64,64,64)
        self.lks2 = RepLKSSMBlock(64)

        # self.lks3 = RepLKSSMBlock(64)
        self.cls = nn.Sequential(ChannelSSM(64,64),nn.Conv2d(64, num_cls, 7, padding=3))

        # for param in self.encoder.parameters():
        #     param.requires_grad = False

        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x1):
        f1_list = self.encoder(x1)
        
        f4, f2 = f1_list[3], f1_list[1]
        f4 = self.conv1(f4)
        f4 = self.lks1(f4)
        f4 = self.conv11(f4)
        f4 = F.interpolate(f4, f2.shape[-2:], mode='bilinear')

        f2 = self.conv2(f2)
    
        f = self.df(f4, f2)
        f = self.lks2(f)
        f = F.interpolate(f, scale_factor=8, mode='bilinear')
        # f = self.lks3(f)
        f = self.cls(f)
        # f = F.softmax(f)
        return f


class Decoder_seg(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up1 = UpConvBlock(64,64,4)
        self.ssm1 = DTMS(64,64)
        self.up2 = UpConvBlock(64,64,8)
    
    def forward(self, x1, x2):
        f4 = self.up1(x2)
        f2 = self.ssm1(x1, f4)
        f2 = self.up2(f2)
        return f2
