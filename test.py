import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import glob
import argparse
from einops import rearrange, repeat
import imageio
from cd_models.samcd.sam_cd import SAM_CD
from cd_models.samcd.ucd_scm import UCD_SCM
from cd_models.unet_pytorch import UNet
from cd_models.scd_sam import SCD_SAM, SCD_SAM_BCD
from cd_models.cienet import CIENet_VMB
from cd_models.eafhnet import EAFHNet
from cd_models.samcd import Meta_CD
from cd_models.afcf3dnet import AFCF3D
from cd_models.mfnet import MFNet


# pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
# m = FGFPVM_Seg(img_size=512, num_cls=7).cuda()
m = MFNet().cuda(1)
x = torch.randn(1, 3, 512, 512).cuda(1)
y = m(x, x)
print(y.shape)