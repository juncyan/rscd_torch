import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vmamba.vmamba import Permute, SS2D, VSSBlock
from .utils import ConvBNAct, DecomposedConv, BNAct


class MISFM(nn.Module):
    #Multi-scale Information Spatio-temporal Fusion Module
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.zip = ConvBNAct(in_channels*2, in_channels, 3, padding=1)
        self.cbr = ConvBNAct(in_channels, out_channels, 1)
        self.msfa = MSFA(out_channels)
        self.resss2d = ResSS2D(out_channels)
        self.out2 = ConvBNAct(out_channels, out_channels, 3, padding=1)

    
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], 1)
        x = self.zip(x)
        y = self.cbr(x)
        y = self.msfa(y)
        y = self.resss2d(y)
        y = self.out2(y)
        return y

class MSFA(nn.Module):
    #Multi-Scale Feature Aggregation
    def __init__(self, dims):
        super().__init__()
        self.dconv1 = DecomposedConv(dims, dims, 7)
        self.dconv2 = DecomposedConv(dims, dims, 5)
        self.dwc2 = nn.Conv2d(dims, dims, 3, padding=1, groups=dims)
        self.dwc3 = nn.Conv2d(dims, dims, 1, groups=dims)
        self.ba = ConvBNAct(dims*4, dims)#BNAct(dims)
        self.out1 = ConvBNAct(dims, dims, 1)
    
    def forward(self, x):
        y = x
        y1 = self.dconv1(y)
        y2 = self.dconv1(y)
        y3 = self.dwc2(y)
        y4 = self.dwc3(y)

        y = torch.concat([y1,y2,y3,y4], 1)
        y = self.ba(y)
        y = self.out1(y)
        return y
    
class ResSS2D(nn.Module):
    def __init__(self, dims, use_checkpoint=None):
        super().__init__()
        self.ssm = VSSBlock(hidden_dim=dims, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
                forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                gmlp=False, use_checkpoint=use_checkpoint)
    
    def forward(self, x):
        y = x
        z = y.permute(0,2,3,1)
        z = self.ssm(z)
        z = z.permute(0,3,1,2)
        y = y + z
        return y
    

class ResSS2D_R1(nn.Module):
    def __init__(self, dims, use_checkpoint=None):
        super().__init__()
        self.ssm = VSSBlock(hidden_dim=dims, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
                forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                gmlp=False, use_checkpoint=use_checkpoint)
    
    def forward(self, x):
        y = x
        z = y.permute(0,2,3,1)
        z = self.ssm(z)
        z = z.permute(0,3,1,2)
        return z
