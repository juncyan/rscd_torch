import torch
from cd_models.bisrnet import BiSRNet

x = torch.rand([2,3,256,256]).cuda()
m = BiSRNet().cuda()
y = m(x, x)
for i in y:
    print(i.shape)