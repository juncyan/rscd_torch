import torch

from cd_models.dgma2net import DGMAANet



x = torch.rand([1,3,512,512]).cuda(1)
# m = net_factory("unet_ds", 3,5, 512).cuda()
m = DGMAANet(3,2).cuda(1)

y = m(x, x)
print(y.shape)

