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
from models.model import SAM_Mamba, LargeMamba
from models.replk import SS2Dv_Lark, VSSBlock
import argparse
import timm
from cd_models.unireplknet import UniRepLKNet, unireplknet_b, unireplknet_s

x = torch.rand([1,512,512,16]).cuda(1)
m = VSSBlock(16, forward_type='v3noz').cuda(1)
y = m(x)
print(y.shape)