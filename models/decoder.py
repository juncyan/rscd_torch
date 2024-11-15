import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_upsample_layer
from .utils import ConvBNAct
from cd_models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from .scconv import ScConv
from .replk import SS2Dv_Lark

class DTMS(nn.Module):
    #Dual token modeling SSM
    def __init__(self, in_ch, out_ch, ssm=VSSBlock, channel_first=True, use_checkpoint=False):
        super().__init__()
        self.channel_first = channel_first
        dim1 = 2*in_ch
        self.ssm1 = ssm(hidden_dim=dim1, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
                forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                gmlp=False, use_checkpoint=use_checkpoint)
        
        self.ssm2 = ssm(hidden_dim= in_ch, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
                ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
                ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
                forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
                gmlp=False, use_checkpoint=use_checkpoint)
        
        # self.ssm3 = ssm(hidden_dim= 4*in_ch, drop_path=0.1, norm_layer=nn.LayerNorm, channel_first=False,
        #         ssm_d_state=1, ssm_ratio=2.0, ssm_dt_rank='auto', ssm_act_layer=nn.SiLU,
        #         ssm_conv=3, ssm_conv_bias=False, ssm_drop_rate=0.0, ssm_init='v0',
        #         forward_type='v3noz', mlp_ratio=4.0, mlp_act_layer=nn.GELU, mlp_drop_rate=0.0,
        #         gmlp=False, use_checkpoint=use_checkpoint)
        

        self.cbr1 = ConvBNAct(4*in_ch, in_ch, 1)
        self.cbr2 = ConvBNAct(in_ch, in_ch,3,1,padding=1)
        self.cbr3 = ConvBNAct(in_ch, out_ch, 1)
        

    def forward(self, x1, x2):
        _device = x1.device
        B, H, W, C = x1.size()

        t1 = torch.empty(B, H, W, 2*C).cuda(_device)
        # t1 = torch.cat([x1, x2], dim=-1)
        t1[:,:,:, 0::2] = x1
        t1[:,:,:, 1::2] = x2
        
        t1 = self.ssm1(t1)  

        t2 = torch.empty(B, H, 2*W, C).cuda(_device)
        # x1 = self.cbr(x1)
        t2[:, :, 0::2, :] = x1
        t2[:, :, 1::2, :] = x2
        # # t2 = t2.permute(0,2,3,1)
        # t2 = torch.cat([x1, x2], 2)
        t2 = self.ssm2(t2)
        
        t = torch.cat([t1, t2[:, :, :W, :], t2[:, :, W: , :]], dim=-1)
        # t = self.ssm3(t)
        t = t.permute(0, 3, 1, 2)
        t = self.cbr1(t)
        t = self.cbr2(t)
        t = self.cbr3(t)
        return t


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class UpConvBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor = 4,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super().__init__()
        skip_channels = out_channels
        self.scale_factor = scale_factor
        self.con1 = ConvModule(
                in_channels,
                skip_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        
        self.con2 = SS2Dv_Lark(skip_channels, out_channels)

    def forward(self, x):
        """Forward function."""

        y = F.interpolate(x, scale_factor=self.scale_factor,mode = 'bilinear')
        y = self.con1(y)
        y = self.con2(y)

        return y