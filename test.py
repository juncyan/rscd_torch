import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from cd_models.scd_sam.scd_sam import SCD_SAM
from scdmodel.emamba import MambaLayer, EM_UNet
from scdmodel.lkm import LKSSMBlock
from scdmodel.sam_mamba import SCDSamMamba, SCDSamMambaLK
from scdmodel.adaptive_sam import MSAMMamba
from cd_models.isdanet import ISDANet




if __name__ == "__main__":
    print('test')
    m = ISDANet(3).cuda()
    x = torch.randn(4,3,256,256).cuda()
    l = torch.argmax(x, dim=1).cuda()
    y = m(x, x)
    loss = m.loss(y, l)
    p = m.predict(y)
    print(p.shape)
    print(loss)
    for i in y:
        print(i.shape)