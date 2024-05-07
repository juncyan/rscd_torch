import torch

import torch.nn as nn

torch.cuda.set_device(1)

x = torch.rand([1,3,244,244]).cuda()
m = nn.Conv2d(3,1,3).cuda()

y = m(x)