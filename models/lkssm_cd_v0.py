import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from cd_models.unireplknet import  unireplknet_s, unireplknet_b
from timm.models.helpers import named_apply

from .utils import ConvBNAct, _init_weights
from .replk import SS2D_v3, DilatedReparamBlock


# RepLK Convolutional Additive Mamba for Land Cover Fain-graind Understanding

class RepLKSSM_CD_v0(nn.Module):
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
        
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

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


class UpConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor = 4
                 ):
        super().__init__()
     
        self.scale_factor = scale_factor
        self.skip = ConvBNAct(in_channels, out_channels, 1)

        self.con1 = ConvBNAct(in_channels, out_channels, 1)
        self.dlk = DilatedReparamBlock(out_channels, 13)
        self.ssm3 = SS2D_v3(out_channels, out_channels)
        self.ssm4 = SS2D_v3(out_channels, out_channels)

    def forward(self, x):
        y2 = self.skip(x)

        y1 = self.con1(x).contiguous()
        y1 = self.dlk(y1)
        y1 = self.ssm3(y1)
        y1 = self.ssm4(y1)
        
        y = y1 + y2
        
        y = F.interpolate(y, scale_factor=self.scale_factor,mode = 'bilinear')
        
        return y


class Decoder(nn.Module):
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


class DTMS(nn.Module):
    #Dual token modeling SSM
    def __init__(self, in_ch, out_ch, channel_first=True):
        super().__init__()
        self.channel_first = channel_first
        dim1 = 2*in_ch
        self.ssm1 = SS2D_v3(dim1, out_ch)
        self.ssm2 = SS2D_v3(in_ch, out_ch)
        # self.ssm3 = ssm(hidden_dim= 4*in_ch, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
        #         ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
        #         ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
        #         forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
        #         gmlp=False, use_checkpoint=use_checkpoint)
        self.conv = ConvBNAct(2*in_ch, out_ch, 1)
        self.cbr1 = ConvBNAct(3*out_ch, out_ch, 1)
        self.cbr2 = DilatedReparamBlock(out_ch, 13)
        self.ssm3 = SS2D_v3(out_ch, out_ch)
        self.cbr3 = SS2D_v3(out_ch, out_ch)
        

    def forward(self, x1, x2):
        _device = x1.device
        B, C, H, W = x1.size()
        tp = torch.cat([x1, x2], dim=1)
        tp = self.conv(tp)

        t1 = torch.empty(B, 2*C, H, W).cuda(_device)
        # t1 = torch.cat([x1, x2], dim=-1)
        t1[:, 0::2, :, :] = x1
        t1[:, 1::2, :, :] = x2
        
        t1 = self.ssm1(t1)  

        t2 = torch.empty(B, C, H, 2*W).cuda(_device)
        # x1 = self.cbr(x1)
        t2[:, :, :, 0::2] = x1
        t2[:, :, :, 1::2] = x2
        # # t2 = t2.permute(0,2,3,1)
        # t2 = torch.cat([x1, x2], 2)
        t2 = self.ssm2(t2)
        
        t = torch.cat([t1, t2[:, :, :, :W], t2[:, :, :, W:]], dim=1).contiguous()
        # t = self.ssm3(t)
        # t = t.permute(0, 3, 1, 2)
        t = self.cbr1(t)
        t = self.cbr2(t)
        t = self.ssm3(t)
        t = self.cbr3(t)
        
        t = t + tp
        
        return t