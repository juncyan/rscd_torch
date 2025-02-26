import torch
import torch.nn as nn
from torch.nn import functional as F
from cd_models.mobilesam import build_sam_vit_t
from cd_models import layers

from .cddecoder import CD_Mamba, CD_CrossA
from .decoder import SemantiMambacDv0, SemantiCrossA
from .utils import features_transfer


class SCDSam_CrossA(nn.Module):
    def __init__(self, img_size=256,num_seg=7,num_cd=2):
        super().__init__()
        self.sam = build_sam_vit_t(img_size=img_size)
        self.sam.eval()

        self.fusion1 = SemantiCrossA(320,128,64,img_size)

        self.cdfusion = CD_CrossA(320,128,64,img_size)

        self.cls = layers.ConvBN(64,1,7)
        self.scls1 = layers.ConvBN(64,num_seg,7)
        self.scls2 = layers.ConvBN(64,num_seg,7)

        for param in self.sam.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        if x2 is None:
            x2 = x1[:,3:,:,:]
            x1 = x1[:,:3,:,:]
            
        b1, b4, b, p1, p4, p = self.extractor(x1, x2)
        tb1 = self.fusion1(b1, b4, b)
        tp2 = self.fusion1(p1, p4, p)
        
        t= self.cdfusion(b1, b4, p1, b4)

        t = self.cls(t)
        outa = self.scls1(tb1)
        outb = self.scls1(tp2)
        return t, outa, outb
    
    # def feature_extractor(self, x):
    #     f1, f2,f3,f4, f = self.sam(x)
    #     return f1, f2, f3, f4, f
    
    def extractor(self, x1, x2):
        b1, b2, b3, b4, b = self.sam(x1)
        p1, p2, p3, p4, p = self.sam(x2)
        # print(b1.shape, b2.shape, b3.shape, b4.shape, p1.shape, p2.shape, p3.shape, p4.shape)

        return b1, b3, b, p1, p3, p