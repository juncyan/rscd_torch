# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
from skimage import io
import os
import pandas as pd
from cd_models.scd_sam import SCD_SAM

x1 = torch.rand(1,3,512,512).cuda()
m = SCD_SAM().cuda()

y = m(x1, x1)
for i in y:
    print(i.shape)