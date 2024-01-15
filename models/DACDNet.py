import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvBNReLU
from ..cd_models.utils import *

class SFF(nn.Module):
    def __init__(self, in_channels, out_size=[32, 32], blocks=[1, 2, 4, 6], pool=F.adaptive_max_pool2d):
        super().__init__()
        assert type(out_size) == int or type(out_size) == list, "output size illeged"
        if type(out_size) == list:
            self.out_size = out_size
        else:
            self.out_size = [out_size, out_size]
        self.pool = pool
        self.blocks = blocks
        num_blocks = len(self.blocks)
        out_channels = max(1, in_channels // num_blocks)
        cbrs = [ConvBNReLU(in_channels, out_channels, 1, stride=1, padding=0)] * num_blocks
        self.cbrs = nn.ModuleList(cbrs)

    def forward(self, x):
        _, _, w, h = x.shape
        max_level = max(self.blocks)
        assert w >= max_level and h >= max_level, "input x size smaller than pool size"
        y = []
        for block, layer in zip(self.blocks, self.cbrs):
            t = self.pool(x, block)
            t = layer(t)
            y.append(F.interpolate(t, size=self.out_size, mode='bilinear'))
        y = torch.concat(y, dim=1)
        return y




class CBRGroup(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super(CBRGroup, self).__init__()
        if down:
            self.cbrg = nn.Sequential(
                ConvBnReLU(in_channels, in_channels//2, 3, 1, 1),
                ConvBnReLU(in_channels//2, out_channels, 3, 1, 1))
        else:
            self.cbrg = nn.Sequential(
                ConvBnReLU(in_channels, out_channels//2, 3, 1, 1),
                ConvBnReLU(out_channels//2, out_channels, 3, 1, 1))

    def forward(self, x):
        return self.cbrg(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.act = nn.Sequential()
        self.act.add_module('conv',nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False))
        self.act.add_module('cbrg',CBRGroup(in_channels, out_channels))
    
    def forward(self, x):
        return self.act(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
 
        self.up = CBRGroup(in_channels, out_channels, False)

    def forward(self, x1, x2):
        
        x1 =nn.functional.interpolate(x1,scale_factor=2, mode='bilinear', align_corners=True)
        
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.up(x)


class FEModule(nn.Module):
    def __init__(self, in_channels, mid_channels:list=[32,64,128]):
        super(FEModule, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvBnReLU(in_channels, 64, 1, 1))
        in_channels = 64
        for c in mid_channels:
            self.layers.append(nn.Sequential(nn.Conv2d(in_channels, in_channels, 3,2,1),
                                             CBRGroup(in_channels, c)))
            in_channels = c

    def forward(self, x):
        y = x
        res = []
        for layer in self.layers:
            y = layer(y)
            res.append(y)
        return res

class FDModule(nn.Module):
    # Feature Difference
    def __init__(self, in_channels=3, mid_channels = [32,64,128]):
        super(FDModule, self).__init__()
        self.branch1 = FEModule(in_channels, mid_channels)
        self.branch2 = FEModule(in_channels, mid_channels)
        
    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i-j
            res.append(torch.clip(z, 0.,1.0))
        return res


class FAModule(nn.Module):
    # Feature Assimilation
    def __init__(self, in_channels=3, mid_channels = [128,256,512]):
        super(FAModule, self).__init__()
        self.branch1 = FEModule(in_channels, mid_channels)
        self.branch2 = FEModule(in_channels, mid_channels)
        
    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        y1 = self.branch1(x1)
        y2 = self.branch2(x2)
        res = []
        for i, j in zip(y1, y2):
            z = i+j
            res.append(torch.clip(z, 0.,1.0))
        return res


class DSCDNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(DSCDNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        # self.fam = FAModule(in_channels//2, [128,256,512])
        self.fdm = FDModule(in_channels//2, [128,256,512])

        self.inc = CBRGroup(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 
        # self.down4 = Down(1024, 1024 // factor)
        self.down4 = SFF(512, [64,64], [1,2,4,6])
        self.up1 = Up(1024, 512 // factor)
        self.up2 = Up(512, 256 // factor)
        self.up3 = Up(256, 128 // factor)
        self.up4 = Up(128, 64)
        self.brige4 = ConvBnReLU(1024,512,1,1)
        self.brige3 = ConvBnReLU(512,256,1,1)
        self.brige2 = ConvBnReLU(256,128,1,1)
        self.brige1 = ConvBnReLU(128,64,1,1)
        self.outc = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        fdm1, fdm2, fdm3, fdm4 = self.fdm(x)
    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x4 = torch.cat([x4, fdm4], dim=1)
        x3 = torch.cat([x3, fdm3], dim=1)
        x2 = torch.cat([x2, fdm2], dim=1)
        x1 = torch.cat([x1, fdm1], dim=1)
        
        x4 = self.brige4(x4)
        x3 = self.brige3(x3)
        x2 = self.brige2(x2)
        x1 = self.brige1(x1)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits




if __name__ == "__main__":
    print("DACDNet Difference Assimilation Change Detection Network")
    

