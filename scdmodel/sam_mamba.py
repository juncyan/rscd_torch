import torch
import torch.nn as nn
import torch.nn.functional as F

from cd_models.mobilesam import build_sam_vit_t
from cd_models import layers
from cd_models.vmamba import SS2D
from .adaptive_sam import MSAMMamba

from models.cdfa import ContrastDrivenFeatureAggregation

from .utils import features_transfer

from .mamba_prior import Mamba_Prior, Mamba_LK

class SCDSamMambaLK(nn.Module):
    def __init__(self, img_size=256,num_seg=7,num_cd=2):
        super().__init__()
        self.sam = MSAMMamba(img_size=img_size)
        # self.sam.eval()

        self.ssm1 = Mamba_LK()
        self.fusion1 = SemantiCrossA(320,128,64,img_size)

        self.cdfusion = CD_CrossA(256,128,64,img_size)

        self.cls = layers.ConvBN(64,1,7)
        self.scls1 = layers.ConvBN(64,num_seg,7)
        self.scls2 = layers.ConvBN(64,num_seg,7)

        # for param in self.sam.parameters():
        #     param.requires_grad = False

    def forward(self, x1, x2):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
            
        b1, b2, b3, b4, b = self.sam(x1)
        b1 = features_transfer(b1, 'NCWH')
        b2 = features_transfer(b2, 'NCWH')
        b3 = features_transfer(b3, 'NCWH')
        # print(b1.shape, b2.shape, b3.shape, b.shape)
        b1, b2, b3, b4 = self.ssm1(b1, b2, b3, b)

        p1, p2, p3, p4, p = self.sam(x2)
        p1 = features_transfer(p1, 'NCWH')
        p2 = features_transfer(p2, 'NCWH')
        p3 = features_transfer(p3, 'NCWH')
        p1, p2, p3, p4 = self.ssm1(p1, p2, p3, p)
       
        tb1 = self.fusion1(b1, b3, b4)
        tp2 = self.fusion1(p1, p3, p4)
        
        t= self.cdfusion(b1, b4, p1, p4)

        t = self.cls(t)
        outa = self.scls1(tb1)
        outb = self.scls1(tp2)
        return t, outa, outb


class SCDSamMamba(nn.Module):
    def __init__(self, img_size=256,num_seg=7,num_cd=2):
        super().__init__()
        self.sam = build_sam_vit_t(img_size=img_size)
        self.sam.eval()

        self.ssm1 = Mamba_Prior()
        self.fusion1 = SemantiCrossA(320,128,64,img_size)

        self.cdfusion = CD_CrossA(256,128,64,img_size)

        self.cls = layers.ConvBN(64,1,7)
        self.scls1 = layers.ConvBN(64,num_seg,7)
        self.scls2 = layers.ConvBN(64,num_seg,7)

        for param in self.sam.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
            
        b1, b2, b3, b4, b = self.sam(x1)
        b1 = features_transfer(b1, 'NWHC')
        b2 = features_transfer(b2, 'NWHC')
        b3 = features_transfer(b3, 'NWHC')
        
        b1, b2, b3, b4 = self.ssm1(b1, b2, b3, b)

        p1, p2, p3, p4, p = self.sam(x2)
        p1 = features_transfer(p1, 'NWHC')
        p2 = features_transfer(p2, 'NWHC')
        p3 = features_transfer(p3, 'NWHC')
        p1, p2, p3, p4 = self.ssm1(p1, p2, p3, p)
       
        tb1 = self.fusion1(b1, b3, b4)
        tp2 = self.fusion1(p1, p3, p4)
        
        t= self.cdfusion(b1, b4, p1, p4)

        t = self.cls(t)
        outa = self.scls1(tb1)
        outb = self.scls1(tp2)
        return t, outa, outb


class CD_CrossA(nn.Module):
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        
        self.proj1 = nn.Conv2d(2*in_c1, in_c1, 1)
        self.ssm1 = layers.ConvBNReLU(in_c1, in_c1, 3)
        self.conv1 = nn.Sequential(layers.ConvBNReLU(in_c1, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))
        
        self.proj2 = nn.Conv2d(2*in_c2, in_c2, 1)
        self.ssm2 = layers.ConvBNReLU(in_c2, in_c2, 3)
        self.conv3 = nn.Sequential(layers.ConvBNReLU(in_c2, out_c, 1), layers.ConvBNReLU(out_c, out_c, 3))

        self.conv4 = layers.ConvBNReLU(out_c+out_c, out_c, 1)

    
    def forward(self, x1, x2, y1, y2):
        f1 = torch.concat([x2, y2], dim=1)
        f1 = self.proj1(f1)
        f1 = self.ssm1(f1)
        f1 = self.conv1(f1)
        f1 = F.interpolate(f1, self.img_size, mode='bilinear', align_corners=True)

        f3 = torch.concat([x1, y1], dim=1)
        f3 = self.proj2(f3)
        f3 = self.ssm2(f3)
        f3 = self.conv3(f3)
        f3 = F.interpolate(f3, self.img_size, mode='bilinear', align_corners=True)

        f4 = torch.concat([f3,f1] , dim=1)
        f4 = self.conv4(f4)
        return f4

class SemantiCrossA(nn.Module):
    """ spatial channel attention module"""
    def __init__(self, in_c1, in_c2, out_c,img_size):
        super().__init__()
        self.img_size = [img_size, img_size]
        self.st1conv1 = layers.ConvBNReLU(in_c1, in_c2, 1)
        self.st1conv2 = layers.ConvBNReLU(in_c2, in_c2, 3)
        self.st1conv3 = layers.ConvBNReLU(in_c2, in_c2, 1)

        self.fg = layers.ConvBNReLU(256,in_c2, 1)
        self.ca = ContrastDrivenFeatureAggregation(in_c2, in_c2)
        
        self.st2conv2 = layers.ConvBNReLU(in_c2, out_c, 1)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 3)
        self.st2conv3 = layers.ConvBNReLU(out_c, out_c, 1)

    def forward(self, x1, x2, x):
        f = self.st1conv1(x2)
        f1 = self.st1conv2(f)
        f1 = self.st1conv3(f1)
        
        f_ = self.fg(x)
        f2 = x1
        f1 = F.interpolate(f1, size = f2.shape[-2:], mode='bilinear', align_corners=True)
        f_ = F.interpolate(f_, size = f2.shape[-2:], mode='bilinear', align_corners=True)

        f2 = self.ca(f2, f1, f_)
        f2 = self.st2conv2(f2)
        f2 = self.st2conv3(f2)
        f2 = self.st2conv3(f2)
        f2 = F.interpolate(f2, size=self.img_size, mode='bilinear', align_corners=True)
        
        # f2 = f2 + f1
        return f2 