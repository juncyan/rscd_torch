import re
import time
import math
import numpy as np
from functools import partial
from typing import Optional, Union, Type, List, Tuple, Callable, Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat, einsum

from cd_models.vmamba import SS2D, VSSBlock
from .utils import ConvBNAct, FFN

class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, down_ratio=2):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 1, stride=1, padding=0)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, stride=down_ratio, padding=1, act='silu')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AdaptiveSS2D(nn.Module):   
    def __init__(self, in_chs, out_chs, down_ratio=4):
        super().__init__()
        self.ds = downsample(in_chs, out_chs, down_ratio)
        self.pool = nn.Sequential(
                    nn.Conv2d(out_chs, out_chs, kernel_size=1),
                    nn.BatchNorm2D(out_chs),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2D(1)
                    )
        self.token = nn.Sequential(SS2D(out_chs),
                                   nn.ModuleNorm(out_chs, epsilon=1e-6))
    
    def forward(self, x):
        x = self.ds(x)
        x1 = self.pool(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.token(x)
        x = rearrange(x, 'b h w c -> b c h w')
        y = x + x1
        # y = self.cb(y)
        return y

class Local_Feature_Gather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim_sp = dim // 2

        self.CDilated_1 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=1, dilation=1, groups=self.dim_sp)
        self.CDilated_2 = nn.Conv2d(self.dim_sp, self.dim_sp, 3, stride=1, padding=2, dilation=2, groups=self.dim_sp)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        cd1 = self.CDilated_1(x1)
        cd2 = self.CDilated_2(x2)
        x = torch.concat([cd1, cd2], dim=1)
        return x

class Global_Feature_Gather(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU())
        
        self.token = SS2D(self.dim * 2)
        self.conv_fina = nn.Sequential(
            nn.Conv2d(2*dim, dim, 1),
            nn.GELU()) 

    def forward(self, x):
        x = self.conv_init(x)
        x0 = x
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.token(x)
        x = rearrange(x, 'b h w c -> b c h w')
        
        x = self.conv_fina(x + x0)
        return x

class MultiScaleFeatureGather(nn.Module):
    def __init__(self, dim):
        super(MultiScaleFeatureGather, self).__init__()
        self.dim = dim
        self.conv_init = ConvBNAct(dim, dim * 2, 1, stride=1, padding=0, act='gelu')
        # layers.ConvBNReLU(dim, dim * 2, 1)

        self.mixer_local = Local_Feature_Gather(dim=self.dim)
        self.mixer_gloal = Global_Feature_Gather(dim=self.dim)

        self.ca_conv = nn.Sequential(
            nn.Conv2d(2 * dim, dim, 1),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * dim, 2 * dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(2 * dim // 2, 2 * dim, kernel_size=1),
            nn.Sigmoid())

        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv_init(x)
        x = torch.split(x, self.dim, dim=1)
        
        x_local = self.mixer_local(x[0])
        x_gloal = self.mixer_gloal(x[1])
        x = torch.concat([x_local, x_gloal], dim=1)
        x = self.gelu(x)
        x = self.ca(x) * x
        x = self.ca_conv(x)

        return x

class Decoder(nn.Module):
    def __init__(self, img_size, channels = [64,256]):
        super(Decoder, self).__init__()
        self.img_size = [img_size, img_size]
        # self.mixter11 = GlobalTokenAttention(channels[-1])
        self.mixter1 = MultiScaleFeatureGather(channels[-1])
        # self.conv1 = layers.ConvBNAct(channels[-1], channels[0], 3, act='gelu') #nn.Conv2d(channels[-1], channels[0], 1)

        self.mixter2 = MultiScaleFeatureGather(channels[0])  
        # self.conv2 = layers.ConvBNAct(channels[0], 64, 3, act='gelu') #nn.Conv2d(channels[0], 64, 1)

        self.cdfa = ParallChangeInformationFusion(channels[-1], channels[0], 64)
        self.conv5 = ConvBNAct(64, 64, 3, padding=1)

    def forward(self, x1, x2):
        y5 = self.mixter1(x2)

        y4 = self.mixter2(x1)
        # y4 = self.conv2(y4)
        y4 = self.cdfa(y5, y4)
        y = F.interpolate(y4, size=self.img_size, mode='bilinear', align_corners=True)

        y = self.conv5(y)
        
        return y

class SegDecoder(nn.Module):
    def __init__(self, img_size, channels = [128,160,320,256]):
        super(SegDecoder, self).__init__()
        self.img_size = [img_size, img_size]
        self.mixter1 = MultiScaleFeatureGather(channels[3])
        self.conv1 = nn.Conv2d(channels[3], channels[2], 1)

        self.mixter2 = MultiScaleFeatureGather(channels[2])
        self.conv2 = nn.Conv2d(channels[2], channels[1], 1)

        self.mixter3 = MultiScaleFeatureGather(channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[0], 1)

        self.mixter4 = MultiScaleFeatureGather(channels[0])
        self.conv4 = nn.Conv2d(channels[0], 64, 1)

        self.conv5 = ConvBNAct(64, 64, 3, padding=1)

    def forward(self, f2, f3, f4, f5):
        y5 = self.mixter1(f5)
        y5 = self.conv1(y5)
        # y5 = F.interpolate(y5, size=self.img_size, mode='bilinear', align_corners=True)
        f4 = f4 + y5
        y4 = self.mixter2(f4)
        y4 = self.conv2(y4)
        # y4 = F.interpolate(y5, size=self.img_size, mode='bilinear', align_corners=True)
        f3 = f3 + y4
        y3 = self.mixter3(f3)
        y3 = self.conv3(y3)
        y3 = F.interpolate(y3, scale_factor=2, mode='bilinear', align_corners=True)
        f2 = f2 + y3
        y2 = self.mixter4(f2)
        y2 = self.conv4(y2)
        y2 = F.interpolate(y2, size=self.img_size, mode='bilinear', align_corners=True)

        # y = torch.concat([y2, y3, y4, y5], dim=1)
        y = self.conv5(y2)
        
        return y
    
    # def train(train_loader, network, criterion, optimizer):
    #     losses = np.array([0.])
    #     for batch in train_loader:
    #         source_img = batch['source'].cuda()
    #         target_img = batch['target'].cuda()

    #         pred_img = network(source_img)
    #         label_img = target_img
    #         l3 = criterion(pred_img, label_img)
    #         loss_content = l3

    #         label_fft3 = torch.fft.fft2(label_img, dim=(-2, -1))
    #         label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

    #         pred_fft3 = torch.fft.fft2(pred_img, dim=(-2, -1))
    #         pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

    #         f3 = criterion(pred_fft3, label_fft3)
    #         loss_fft = f3


    #         loss = loss_content + 0.1 * loss_fft
    #         losses.update(loss.item())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     return losses.avg

class PyramidMamba(nn.Module):
    def __init__(self, in_chs=512, dim=128, d_state=16, d_conv=4, expand=2, last_feat_size=16):
        super().__init__()
        pool_scales = self.generate_arithmetic_sequence(1, last_feat_size, last_feat_size // 4)
        self.pool_len = len(pool_scales)
        self.pool_layers = nn.ModuleList()
        self.pool_layers.append(nn.Sequential(
                    nn.Conv2d(in_chs, dim, kernel_size=1),
                    nn.BatchNorm2D(dim),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2D(1)
                    ))
        for pool_scale in pool_scales[1:]:
            self.pool_layers.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2D(pool_scale),
                    nn.Conv2d(in_chs, dim, kernel_size=1),
                    nn.BatchNorm2D(dim),
                    nn.ReLU(),
                    ))
        self.mamba = SS2D(dim* self.pool_len + in_chs)#SS2D(dim* self.pool_len + in_chs)
        self.cbr = ConvBNAct(dim* self.pool_len + in_chs, dim, 1, act='gelu')
       
        # self.mamba = MambaLayer(
        #     d_model=dim*self.pool_len+in_chs,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand # Block expansion factor
        # )

    def forward(self, x): # B, C, H, W
        res = x
        B, C, H, W = res.shape
        ppm_out = [res]
        for p in self.pool_layers:
            pool_out = p(x)
            pool_out = F.interpolate(pool_out, (H, W), mode='bilinear', align_corners=False)
            ppm_out.append(pool_out)
        x = torch.concat(ppm_out, dim=1)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.mamba(x)
        x = rearrange(x, 'b h w c -> b c h w')
        # x = x.transpose(2, 1).view(B, chs, H, W)
        x = self.cbr(x)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence

class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
       
        self.qkv = nn.Conv2d(dim, 3 * dim, 1)
        self.oper_q = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.oper_k = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.proj = nn.Sequential(nn.Conv2d(dim, dim, 3,1, padding=1, groups=dim),
                                  nn.BatchNorm2d(dim))
        self.proj_drop = nn.Dropout(0.)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out

class ParallChangeInformationFusion(nn.Module):
    def __init__(self,in_chs1, in_chs2, out_chs):
        super().__init__()
        
        self.W_g = nn.Conv2d(in_chs1, out_chs, kernel_size=1,stride=1)
        self.W_x = nn.Conv2d(in_chs2, out_chs, kernel_size=1,stride=1)

        self.token = VSSBlock(out_chs)

        self.psi = nn.Sequential(
            nn.ReLU(),
            ConvBNAct(out_chs, 1, 1, act='sigmoid'),
        )
        self.cbr = ConvBNAct(out_chs, out_chs, 3, act='silu')
        

    def forward(self, g, x):
        sz = x.shape[-2:]
        g1 = self.W_g(g)
        g1 = F.interpolate(g1, size=sz, mode='bilinear', align_corners=False)
        x1 = self.W_x(x)

        y = g1 + x1
        # y1 = torch.transpose(y, [0, 2, 3, 1])
        y1 = rearrange(y, 'b c h w -> b h w c')
        y1 = self.token(y1)
        # y1 = torch.transpose(y1, [0, 3, 1, 2])
        y1 = rearrange(y1, 'b h w c -> b c h w')
        y2 = self.psi(y)
        y2 = y2 * x1
        y2 = y2 + y1
        y2 = self.cbr(y2)
        
        return y2

class CoarseInteractiveFeaturesExtraction(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(2*dim, dim, 1)
        self.token = VSSBlock(dim)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.ffn = FFN(dim)
    
    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x = torch.empty([B, 2*C, H, W], dtype=x1.dtype).cuda()
        x[:, 0::2, ...] = x1
        x[:, 1::2, ...] = x2

        x = self.conv1(x)
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.token(x)
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.conv2(x)

        x = x + x1 + x2
        x = self.ffn(x)

        return x

class GlobalTokenAttention(nn.Module):
    def __init__(self, embed_dim,
                 head_num=4,
                 **kwargs):
        super().__init__()
    
        self.embed_dim = embed_dim
        self.head_num = head_num
        self.scale = embed_dim ** -0.5
        
        self.qkv = nn.Conv2d(embed_dim, embed_dim*3, kernel_size=1)
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        self.lepe = nn.Conv2d(embed_dim, embed_dim, 3, padding=1) 
        self.ssm = VSSBlock(embed_dim)
            
    #     self._init_params_()

    # def _init_params_(self):
    #     nn.initializer.XavierNormal(gain=2**-2.5)(self.qkv.weight)
    #     nn.initializer.Constant(0.)(self.qkv.bias)
    #     nn.initializer.XavierNormal(gain=2**-2.5)(self.proj.weight)
    #     nn.initializer.Constant(0.)(self.proj.bias)

    def forward(self, x):
        
        B, C, H, W = x.shape
        
        qkv = self.qkv(x)
        lepe = qkv[:, -C:, ...]
        
        q, k, v = rearrange(qkv, 'b (m n c) h w -> m b h n w c', m=3, n=self.head_num)
        k = rearrange(k, 'b n h w c -> b n h c w')
        attn = (q @ k) * self.scale
        attn = F.softmax(attn, dim=-1)
    
        y = attn @ v
        y = rearrange(y, 'b h n w c -> b h w n c')
        y = torch.reshape(y, [B, H, W, -1])
        
        lepe = self.lepe(lepe)
        lepe = rearrange(lepe, 'b c h w -> b h w c')
        lepe = self.ssm(lepe)

        y = y + lepe
        y = rearrange(y, 'b h w c -> b c h w')
        y = self.proj(y)
        
        return y

class GlobalAttention(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1

        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias_attr=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b h w c')#.permute(0, 2, 3, 1)
        N = H * W
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose([0,1,3,2])) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(0, 3, 1, 2)
        return x
