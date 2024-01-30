import torch
from cd_models.rdpnet import RDPNet

x = torch.rand([2,3,512,512]).cuda()
m = RDPNet(3,2).cuda()
y = m(x, x)
print(y.shape)