import torch
from cd_models.changestar.changestar import ChangeStar_R50

x = torch.rand([2,3,512,512]).cuda()
m = ChangeStar_R50().cuda()
y = m(x, x)
print(y.shape)