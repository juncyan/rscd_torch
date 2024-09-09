# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
import os

# 基础功能
from work.train import train
from cd_models.aernet import AERNet

# 模型导入
test_data1 = torch.rand(2,3,256,256).cuda()
test_data2 = torch.rand(2,3,256,256).cuda()
test_label = torch.randint(0, 2, (2,1,256,256)).cuda()

model = AERNet()
model = model.cuda()
output = model(test_data1,test_data2)
p = model.predict(output)
ls = model.loss(output, test_label)
print(ls)