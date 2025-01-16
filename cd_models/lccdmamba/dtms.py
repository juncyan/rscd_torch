import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vmamba.vmamba import VSSBlock
from .utils import ConvBNAct

class DTMS(nn.Module):
    #Dual token modeling SSM
    def __init__(self, in_ch1, in_ch2, ssm=VSSBlock, channel_first=True, use_checkpoint=False):
        super().__init__()
        self.channel_first = channel_first
        dim1 = in_ch1 + in_ch2
        # self.conv = nn.Conv2d(in_channels=in_channels, out_channels=encoder_dims, kernel_size=1)
        self.ssm1 = ssm(hidden_dim=dim1, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
                forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                gmlp=False, use_checkpoint=use_checkpoint)
        
        self.ssm2 = ssm(hidden_dim=in_ch2, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
                forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                gmlp=False, use_checkpoint=use_checkpoint)
        

        self.cbr = ConvBNAct(in_ch1, in_ch2, 3, padding=1)
        self.cbr2 = ConvBNAct(dim1+2*in_ch2, in_ch2, 3, padding=1)
        self.cbr3 = ConvBNAct(in_ch2, in_ch2, 3, 1, padding=1)
        

    def forward(self, input1, input2):
        _device = input1.device
        B, C, H, W = input2.size()
        x1 = F.upsample_bilinear(input1, size=(H, W))
        x2 = input2

        t1 = torch.cat([x1, x2], dim=1)
        t1 = t1.permute(0,2,3,1)
        # print(t1.shape)
        t1 = self.ssm1(t1)  

        t2 = torch.empty(B, C, H, 2*W).cuda(_device)
        x1 = self.cbr(x1)
        t2[:, :, :, :W] = x1
        t2[:, :, :, W:] = x2
        t2 = t2.permute(0,2,3,1)
        t2 = self.ssm2(t2)
        
        t = torch.cat([t1, t2[:, :, :W, :], t2[:, :, W: , :]], dim=-1)
        t = t.permute(0, 3, 1, 2)
        t = self.cbr2(t)
        t = self.cbr3(t)
        return t
