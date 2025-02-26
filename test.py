import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from cd_models.scd_sam.scd_sam import SCD_SAM
from scdmodel.emamba import MambaLayer, EM_UNet
from scdmodel.lkm import LKSSMBlock
from scdmodel.sam_mamba import SCDSamMamba, SCDSamMambaLK
from scdmodel.adaptive_sam import MSAMMamba




if __name__ == "__main__":
    print('test')
    m = MSAMMamba(256).cuda()
    x = torch.randn(4,3,256,256).cuda()
    y = m(x)
    # for i in y:
    #     print(i.shape)