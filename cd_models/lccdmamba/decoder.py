import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .vmamba.vmamba import VSSBlock
from .utils import ConvBNAct
from .misf import MISFM
from .dtms import DTMS



class Decoder(nn.Module):
    def __init__(self, dims, out_channels=64,channel_first=False):
        super().__init__()
        dim1, dim2, dim3, dim4 = dims
        print(dims)
        self.msf4 = MISFM(dim4)

        self.msf3 = MISFM(dim3)
        self.dts3 = DTMS(128, 128)

        self.msf2 = MISFM(dim2)
        self.dts2 = DTMS(128, 128)
        
        self.msf1 = MISFM(dim1)
        self.dts1 = DTMS(128, 128)

        self.cbr = ConvBNAct(128, out_channels, 3, padding=1)

    
    def forward(self, x1, x2):
        pre1, pre2, pre3, pre4 = x1
        pos1, pos2, pos3, pos4 = x2
        
        f4 = self.msf4(pre4, pos4)
        f3 = self.msf3(pre3, pos3)
        f2 = self.msf2(pre2, pos2)
        f1 = self.msf1(pre1, pos1)

        f = self.dts3(f4, f3)
        f = self.dts2(f, f2)
        f = self.dts1(f, f1)

        f = self.cbr(f)
        
        return f

    
class UpBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2) -> None:
        super().__init__()
        dims =  in_ch1 + in_ch2
        self.conv1 = ConvBNAct(dims, in_ch2, 1)
        self.conv2 = ConvBNAct(in_ch2, in_ch2, 3, padding=1)
        self.conv3 = ConvBNAct(in_ch2, in_ch2, 3, padding=1)

    def forward(self, x1, x2):
        _,_,H, W = x2.shape
        x1 = F.upsample_bilinear(x1, size=(H, W))
        x2 = x2
        y = torch.cat([x1, x2], 1)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)
        return y