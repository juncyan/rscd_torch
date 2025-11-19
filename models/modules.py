import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------
# FFN
# ------------------------
class FFN(nn.Module):
    def __init__(self, dim):
        super(FFN, self).__init__()
        self.dim = dim
        self.dim_sp = dim // 2

        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, 2 * dim, 1),
        )

        self.conv1_1 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=3, padding=1, groups=self.dim_sp)
        self.conv1_2 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=5, padding=2, groups=self.dim_sp)
        self.conv1_3 = nn.Conv2d(self.dim_sp, self.dim_sp, kernel_size=7, padding=3, groups=self.dim_sp)

        self.gelu = nn.GELU()
        self.conv_fina = nn.Sequential(
            nn.Conv2d(self.dim_sp, dim, 1),
        )

    def forward(self, x):
        x = self.conv_init(x)
        x = torch.split(x, self.dim_sp, dim=1)
        x = list(x)
        x[1] = self.conv1_1(x[1])
        x[2] = self.conv1_2(x[2])
        x[3] = self.conv1_3(x[3])

        y = x[0] + x[1] + x[2] + x[3]
        y = self.gelu(y)
        y = self.conv_fina(y)
        return y


# ------------------------
# Fourier
# ------------------------
class Fourier(nn.Module):
    def __init__(self, in_channels, out_channels=None, groups=1):
        super(Fourier, self).__init__()
        self.groups = groups
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Conv2d(in_channels * 2,
                              out_channels * 2,
                              kernel_size=1, stride=1, padding=0,
                              groups=self.groups, bias=False)
        self.relu = nn.GELU()
        self.idea = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        b, _, h, w = x.shape

        avg = F.adaptive_avg_pool2d(x, (1, 1))
        avg = self.idea(avg) * x

        # (b, c, h, w//2 + 1, 2)
        ffted = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = ffted.real.unsqueeze(-1)
        x_fft_imag = ffted.imag.unsqueeze(-1)
        ffted = torch.cat((x_fft_real, x_fft_imag), dim=-1)

        # (b, c, 2, h, w//2 + 1)
        ffted = ffted.permute(0, 1, 4, 2, 3)
        ffted = ffted.reshape(b, -1, *ffted.shape[3:])

        ffted = self.conv(ffted)  # (batch, c*2, h, w//2 + 1)

        ffted = ffted.reshape(b, -1, 2, *ffted.shape[2:]).permute(0, 1, 3, 4, 2)  # (batch, c, h, w//2 + 1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfft2(ffted, s=(h, w), norm='ortho')
        output = output + avg
        output = self.relu(output)
        return output


# ------------------------
# CoarseDifferenceFeaturesExtraction
# ------------------------
class CoarseDifferenceFeaturesExtraction(nn.Module):
    def __init__(self, dim=512):
        super(CoarseDifferenceFeaturesExtraction, self).__init__()
        self.conv1 = nn.Conv2d(2 * dim, dim, 1)
        self.fourier = Fourier(dim)
        self.ffn = FFN(dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        y = self.conv1(x)
        y1 = self.fourier(y)
        y1 = y1 + y
        y2 = self.ffn(y1)
        return y2


# ------------------------
# FrequencyDomainFeatureEnhance
# ------------------------
class FrequencyDomainFeatureEnhance(nn.Module):
    def __init__(self, inc, dim, outc):
        super(FrequencyDomainFeatureEnhance, self).__init__()
        self.conv = nn.Conv2d(inc, dim, 1)
        self.fft = Fourier(dim)
        self.conv2 = nn.ConvTranspose2d(
            in_channels=dim,
            out_channels=outc,
            kernel_size=2,
            stride=2
        )
        self.ffn = FFN(outc)

    def forward(self, x):
        x = self.conv(x)
        y = self.fft(x)
        y = self.conv2(y)
        y = self.ffn(y)
        return y


# ------------------------
# SEBlock (Squeeze and Excitation Block)
# ------------------------
class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(input_channels, internal_neurons, 1)
        self.up = nn.Conv2d(internal_neurons, input_channels, 1)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.shape[3])
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.reshape([-1, self.input_channels, 1, 1])
        return inputs * x
