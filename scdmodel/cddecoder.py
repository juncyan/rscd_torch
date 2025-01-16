import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import features_transfer
from cd_models import layers

from cd_models.vmamba import SS2D
from cd_models import layers
from models.cdfa import ContrastDrivenFeatureAggregation


class CD_Mamba(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        # cfg1 = MambaConfig(4096, in_c1)
        # cfg2 = MambaConfig(1024, in_c2)
        # self.ssm1 = nn.Sequential(MambaMixer(cfg1, 0), MambaMixer(cfg1, in_c1-1), nn.LayerNorm(in_c1))
        # self.ssm2 = nn.Sequential(MambaMixer(cfg2, 0), MambaMixer(cfg2, in_c2-1), nn.LayerNorm(in_c2))
        
        self.proj1 = nn.Linear(2*in_c1, in_c1)
        self.ssm1 = SS2D(in_c1)
        self.conv1 = nn.Sequential(layers.ConvBNReLU(in_c1, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))
        
        self.proj2 = nn.Sequential(nn.Linear(2*in_c2, in_c2), nn.LayerNorm(in_c2))
        self.ssm2 = SS2D(in_c2)
        self.conv3 = nn.Sequential(layers.ConvBNReLU(in_c2, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))

        self.conv4 = layers.ConvBNReLU(out_c+out_c, out_c, 1)

    
    def forward(self, x1, x2, y1, y2):
        f1 = torch.concat([x2, y2], dim=-1)
        f1 = self.proj1(f1)
        f1 = features_transfer(f1, "NWHC")
        f1 = self.ssm1(f1)
        f1 = f1.permute(0, 3, 1, 2)
        f1 = self.conv1(f1)
        f1 = F.interpolate(f1, self.img_size, mode='bilinear', align_corners=True)

        f3 = torch.concat([x1, y1], dim=-1)
        f3 = self.proj2(f3)
        f3 = features_transfer(f3, 'NWHC')
        f3 = self.ssm2(f3)
        f3 = f3.permute(0, 3, 1, 2)
        f3 = self.conv3(f3)
        f3 = F.interpolate(f3, self.img_size, mode='bilinear', align_corners=True)

        f4 = torch.concat([f3,f1] , dim=1)
        f4 = self.conv4(f4)
        return f4


class CD_CrossA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
       
        self.proj1 = layers.ConvBNReLU(2*in_c1, in_c1, 1)
        self.conv11 = layers.ConvBNReLU(in_c1, out_c, 1)
        self.conv12 = layers.ConvBNReLU(in_c1, out_c, 1)
        
        self.proj2 = layers.ConvBNReLU(2*in_c2, in_c2, 1)
        self.conv3 = layers.ConvBNReLU(in_c2, 2*out_c, 1)
        self.ac = ContrastDrivenFeatureAggregation(2*out_c, out_c)
        self.conv4 = layers.ConvBNReLU(out_c, out_c, 1)

    
    def forward(self, x1, x2, y1, y2):
        f1 = torch.concat([x1, y1], dim=1)
        
        f1 = self.proj1(f1)
        f1 = F.interpolate(f1, self.img_size, mode='bilinear', align_corners=True)
        f11 = self.conv11(f1)
        f12 = self.conv12(f1)
        
        f3 = torch.concat([x2, y2], dim=1)
        f3 = self.proj2(f3)
        f3 = self.conv3(f3)
        # f3 = F.interpolate(f3, self.img_size, mode='bilinear', align_corners=True)
        f4 = self.ac(f3, f11, f12)
        f4 = self.conv4(f4)
        return f4
    