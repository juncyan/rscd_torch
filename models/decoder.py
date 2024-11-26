import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ConvBNAct
# from cd_models.vmamba import VSSM, LayerNorm2d, VSSBlock, Permute
from .replk import SS2D_v3, DilatedReparamBlock, LKSSMBlock
from .ram import AdditiveTokenMixer



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


class DTMS_v1(nn.Module):
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
        # self.cbr2 = AdditiveTokenMixer(out_ch)
        self.cbr3 = ConvBNAct(out_ch, out_ch, 3, padding=1)

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
        # t = self.cbr2(t)
        t = self.cbr3(t)
        
        t = t + tp
        
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
    

class UpConvBlock_v1(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor = 4
                 ):
        super().__init__()
     
        self.scale_factor = scale_factor
        self.skip = ConvBNAct(in_channels, out_channels, 1)

        self.cov1 = ConvBNAct(in_channels, out_channels, 1)
        self.dlk1 = AdditiveTokenMixer(out_channels)
        self.cov2 = ConvBNAct(out_channels, out_channels, 3, padding=1)
        # self.dlk2 = AdditiveTokenMixer(out_channels)

    def forward(self, x):
        y2 = self.skip(x)

        y1 = self.cov1(x).contiguous()
        y1 = self.dlk1(y1)
        y1 = self.cov2(y1)
        
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


class Decoder_v1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up1 = UpConvBlock_v1(64,64,4)
        self.ssm1 = DTMS_v1(64,64)
        self.up2 = UpConvBlock_v1(64,64,8)
    
    def forward(self, x1, x2):
        f4 = self.up1(x2)
        f2 = self.ssm1(x1, f4)
        f2 = self.up2(f2)
        return f2