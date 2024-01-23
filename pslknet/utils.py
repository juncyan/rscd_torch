import torch
import torch.nn as nn

class DepthWiseConv2D(nn.Module):
    def __init__(self, in_channels, kernel, stride, bias=True):
        super(DepthWiseConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel, stride=stride, padding=kernel//2, groups=in_channels, bias=bias)

    def forward(self, x):
        y = self.conv(x)
        return y


class SeparableConvBNReLU(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride, pointwise_bias=True):
        super().__init__()
        self.depthwise_conv = nn.Sequential(DepthWiseConv2D(in_channels, kernel_size, stride=stride),
                                           nn.BatchNorm2d(in_channels))

        self.piontwise_conv = ConvBnReLU(in_channels,out_channels,kernel_size=1,stride=1,bias=pointwise_bias)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.piontwise_conv(x)
        return x


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y

class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,stride, padding, dilation, groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        return y


class SEModule(nn.Module):
    def __init__(self, channels, reductions = 8):
        super(SEModule, self).__init__()
        reduction_channels = channels // reductions
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduction_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(reduction_channels, channels, 1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # b, c , _, _ = x.shape
        avg = self.avg(x)
        y = self.fc1(avg)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sig(y)
        y = y.expand_as(x)
        return x * y


class PAM(nn.Module):
    """
    Position attention module.
    Args:
        in_channels (int): The number of input channels.
    """

    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 8
        self.mid_channels = mid_channels
        self.in_channels = in_channels

        self.query_conv = nn.Conv2d(in_channels, mid_channels, 1, 1)
        self.key_conv = nn.Conv2d(in_channels, mid_channels, 1, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, 1)

        self.gamma = self.create_parameter(
            shape=[1],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0))

    def forward(self, x):
        x_shape = torch.shape(x)

        # query: n, h * w, c1
        query = self.query_conv(x)
        query = torch.reshape(query, (0, self.mid_channels, -1))
        query = torch.transpose(query, (0, 2, 1))

        # key: n, c1, h * w
        key = self.key_conv(x)
        key = torch.reshape(key, (0, self.mid_channels, -1))

        # sim: n, h * w, h * w
        sim = torch.bmm(query, key)
        sim = F.softmax(sim, axis=-1)

        value = self.value_conv(x)
        value = torch.reshape(value, (0, self.in_channels, -1))
        sim = torch.transpose(sim, (0, 2, 1))

        # feat: from (n, c2, h * w) -> (n, c2, h, w)
        feat = torch.bmm(value, sim)
        feat = torch.reshape(feat,
                              (0, self.in_channels, x_shape[2], x_shape[3]))

        out = self.gamma * feat + x
        return out