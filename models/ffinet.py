import torch
import math
import torch.nn as nn
import torch.nn.functional as F


from .mobilesam import build_sam_vit_t
from .modules import CoarseDifferenceFeaturesExtraction
from .modules import FrequencyDomainFeatureEnhance

class FFINetTV_BCD(nn.Module):
    #Frequency-domain Feature Inteaction Network
    def __init__(self, img_size, num_cls=2, rank=8):
        super().__init__()
        self.img_size = [img_size, img_size]

        self.sam = build_sam_vit_t(img_size=img_size, rank=rank)
        
        self.cife = CoarseDifferenceFeaturesExtraction(256)
        self.upc1 = FrequencyDomainFeatureEnhance(256, 128, 64)
        self.cls = nn.Conv2d(64, num_cls, 3, 1, 1)

        for name, param in self.sam.named_parameters():
            if "lora" in name:
                param.stop_gradient = False
            # else:
            #     param.requires_grad = True
     
    def forward(self, x1, x2=None):
        if x2 is None:
            x = torch.split(x1, 3, dim=1)
            x1 = x[0]
            x2 = x[1]
    
        f, p = self.sam.image_encoder(x1, x2)
        # self.sam.decoder(f)
        
        y = self.cife(f, p)

        y = self.upc1(y)
       
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)
       
        return y

class FFINetTV_SCD(FFINetTV_BCD):
    def __init__(self, img_size, num_seg=7):
        super().__init__(img_size=img_size, num_cls=1)

        self.img_size = [img_size, img_size]
        
        self.up = FrequencyDomainFeatureEnhance(256, 128, 64)
        self.scls1 = nn.Conv2d(64, num_seg, 3, 1, 1)
    
    def forward(self, x1, x2=None):
        if x2 is None:
            x = torch.split(x1, 3, dim=1)
            x1 = x[0]
            x2 = x[1]
    
        f, p = self.sam.image_encoder(x1, x2)

        y = self.cife(f, p)
        y = self.upc1(y)
       
        y = F.interpolate(y, size=self.img_size, mode='bilinear', align_corners=True)
        y = self.cls(y)
        
        s1 = self.up(f)
        s2 = self.up(p)
        s1 = F.interpolate(s1, size=self.img_size, mode='bilinear', align_corners=True)
        s2 = F.interpolate(s2, size=self.img_size, mode='bilinear', align_corners=True)
        s1 = self.scls1(s1)
        s2 = self.scls1(s2)
        return y, s1, s2
        