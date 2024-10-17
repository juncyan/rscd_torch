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

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A  = np.array([48.30,  46.27,  48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B  = np.array([49.41,  47.01,  47.94])

label_inf = pd.read_csv(os.path.join(r"/mnt/data/Datasets/Second/", 'label_info.csv'))

def one_hot_it(label, label_info=label_inf):
    semantic_map = []
    for info in label_info:
        color = label_info[info].values
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    # print(semantic_map)
    return np.stack(semantic_map, axis=-1)

img_dir = r"/mnt/data/Datasets/Second/train/"

l1p = os.path.join(img_dir, 'A')
fs = os.listdir(l1p)
for f in fs:
    i1 = os.path.join(l1p, f)
    img1 = io.imread(i1)
    print(img1.shape)
#     break
# print(len(fs))