import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="same", **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._bn = nn.BatchNorm2d(out_channels)
        self._relu = nn.ReLU6()
    
    def forward(self, x):
        y = self._conv(x)
        y = self._bn(y)
        y = self._relu(y)
        return y


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding="same", **kwargs):
        super().__init__()
        self._conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, **kwargs)
        self._bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        y = self._conv(x)
        y = self._bn(y)
        return y