import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from einops import rearrange, repeat
from cd_models.lccdmamba.configs.config import get_config

from cd_models.scd_sam.scd_sam import SCD_SAM
# from scdmodel.emamba import MambaLayer, EM_UNet
# from scdmodel.lkm import LKSSMBlock
# from scdmodel.sam_mamba import SCDSamMamba, SCDSamMambaLK
# from scdmodel.adaptive_sam import MSAMMamba
from cd_models.isdanet import ISDANet

# from cd_models.vmamba.mamba_backbone import Backbone_VSSM
# from cdmamba.model import VMamba_CD
from cd_models.segman import SegMANEncoder_t, SegMANEncoder_s, SegMANEncoder_b
from cd_models.lwganet.lwunetformer import UNetFormer_lwganet_l0

os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(1)
x = torch.randn(1, 3, 256, 256).cuda(1)
m = UNetFormer_lwganet_l0().cuda(1)
y = m(x)
for i in y:
    print(i.shape)