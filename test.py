import numpy as np
import torch
import argparse
from imageio.v3 import imread
from thop import profile

from models.ffinet import FFINetTV_BCD

# m = FFINetTV_BCD().cuda()
# m.eval()
x = torch.randn(1, 3, 256, 256).cuda()
# y = m(x, x)
# for i in y:
#     print(i.shape)
model = FFINetTV_BCD(256, rank=4)
model = model.cuda()
torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
with torch.no_grad():
    output = model(x, x)

peak_mem = torch.cuda.max_memory_allocated()
print(f"峰值显存占用: {peak_mem / 1024**2:.2f} MB")

flops, params = profile(model, [x,x])
print(f"[PREDICT] model flops is {flops} G, params is {params} M")

