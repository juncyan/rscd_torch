import torch
from pslknet.model import PSLKNet


x = torch.randn([2,3,512,512]).cuda()
m = PSLKNet().cuda()
y = m(x, x)
print(y.shape)