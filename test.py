import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from einops import rearrange, repeat
import imageio
from cd_models.samcd.SAM_CD import SAM_CD
# from cd_models.lccdmamba.configs.config import get_config
# from cd_models.ussfcnet.ussfcnet import USSFCNet
# from cd_models.SNUNet import SNUNet
# from cd_models.scd_sam.scd_sam import SCD_SAM
# # from scdmodel.emamba import MambaLayer, EM_UNet
# # from scdmodel.lkm import LKSSMBlock
# # from scdmodel.sam_mamba import SCDSamMamba, SCDSamMambaLK
# # from scdmodel.adaptive_sam import MSAMMamba
# from cd_models.isdanet import ISDANet

# # from cd_models.vmamba.mamba_backbone import Backbone_VSSM
# # from cdmamba.model import VMamba_CD
# from cd_models.segman import SegMANEncoder_t, SegMANEncoder_s, SegMANEncoder_b
# from cd_models.lwganet.lwunetformer import UNetFormer_lwganet_l0
# from cd_models.lwganet.lwclafr import CLAFR_LWGA
# from cd_models.lwganet.lwa2net import A2Net_LWGANet_L2

# os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(0)
# x = torch.randn(1, 3, 256, 256).cuda(0)
# m = SNUNet(3,2).cuda(0)
# y = m(x, x)
# print(y.shape)
pth = r"/mnt/data/Datasets/Second/train/labelB"
imgs = os.listdir(pth)
for f in imgs:
    img = os.path.join(pth, f)
    img = imageio.imread(img)
    if img.max() == 6:
        print(f)
    # print(f, img.shape, img.max(), img.min())
# x = torch.rand([1,3,256,256]).cuda(1)
# m = SAM_CD(256, device="cuda:1").cuda(1)

# y = m(x,x)
# print(m.predict(y).shape)
