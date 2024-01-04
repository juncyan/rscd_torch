import json
import math
import platform
import warnings
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F




thresh = 0.5  # 0.5 # neuronal threshold
lens = 0.5  # 0.5 # hyper-parameters of approximate function
decay = 0.25  # 0.25 # decay constants
time_window = 1


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        temp = temp / (2 * lens)#？
        return grad_input * temp.float()

act_fun = ActFun.apply

class mem_update(nn.Module):
    def __init__(self,act=False):
        super(mem_update, self).__init__()
        # self.actFun= torch.nn.LeakyReLU(0.2, inplace=False)
        self.actFun = nn.SiLU()
        self.act=act

    def forward(self, x):
        mem = torch.zeros_like(x[0]).to(x.device)
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        for i in range(time_window):
            if i >= 1:
                mem = mem_old * decay * (1-spike.detach()) + x[i]                
            else:
                mem = x[i]
            if self.act:
                spike = self.actFun(mem)
            else:
                spike = act_fun(mem)
                
            mem_old = mem.clone()
            output[i] = spike
        # print(output[0][0][0][0])
        return output



class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        self.act = mem_update(act=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class Conv_A(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class Conv_1(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        #self.act = mem_update() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_fuse(self, x):
        return self.conv(x)
    
class Conv_2(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k, s, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Snn_Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = batch_norm_2d(c2)
        #self.act = mem_update() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.bn(self.conv(x))

    def forward_fuse(self, x):
        return self.conv(x)


class Snn_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', marker='b'):
        super(Snn_Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.marker = marker

    def forward(self, input):
   
        weight = self.weight#
        # print(self.padding[0],'=======')
        h = (input.size()[3]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        w = (input.size()[4]-self.kernel_size[0]+2*self.padding[0])//self.stride[0]+1
        c1 = torch.zeros(time_window, input.size()[1], self.out_channels, h, w, device=input.device)
        # print(weight.size(),'=====weight====')
        for i in range(time_window):
            c1[i] = F.conv2d(input[i], weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return c1

 
class batch_norm_2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d, self).__init__() #num_features=16
        self.bn = BatchNorm3d1(num_features)  # input (N,C,D,H,W) imension batch norm on (N,D,H,W) slice. spatio-temporal Batch Normalization

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)  # 
    
class batch_norm_2d1(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(batch_norm_2d1, self).__init__()
        self.bn = BatchNorm3d2(num_features)

    def forward(self, input):
        y = input.transpose(0, 2).contiguous().transpose(0, 1).contiguous()
        y = self.bn(y)
        return y.contiguous().transpose(0, 1).contiguous().transpose(0, 2)


class BatchNorm3d1(torch.nn.BatchNorm3d):#5
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight, thresh)#
            nn.init.zeros_(self.bias)

class BatchNorm3d2(torch.nn.BatchNorm3d):
    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
  
            nn.init.constant_(self.weight, 0.2*thresh)           
            nn.init.zeros_(self.bias)

class Pools(nn.Module):
    def __init__(self,kernel_size,stride,padding=0,dilation=1):
        super().__init__()
        self.kernel_size=kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.pool=nn.MaxPool2d(kernel_size=self.kernel_size,stride=self.stride,padding=self.padding)

    def forward(self,input):
        h=int((input.size()[3]+2*self.padding-self.dilation*(self.kernel_size-1)-1)/self.stride+1)
        w=int((input.size()[4]+2*self.padding - self.dilation*(self.kernel_size-1)-1)/self.stride+1)
        c1 = torch.zeros(time_window, input.size()[1],input.size()[2],h,w,device=input.device)
        for i in range(time_window):
            c1[i]=self.pool(input[i])
        return c1

class zeropad(nn.Module):
    def __init__(self,padding):
        super().__init__()
        self.padding=padding
        self.pad=nn.ZeroPad2d(padding=self.padding)
    def forward(self,input):
        h=input.size()[3]+self.padding[2]+self.padding[3]
        w=input.size()[4]+self.padding[0]+self.padding[1]
        c1=torch.zeros(time_window,input.size()[1],input.size()[2],h,w,device=input.device )
        for i in range(time_window):
            c1[i]=self.pad(input[i])
        return c1 


class Sample(nn.Module):
    def __init__(self,size=None,scale_factor=None,mode='nearset'):
        super(Sample, self).__init__()
        self.scale_factor=scale_factor
        self.mode=mode
        self.size = size
        self.up=nn.Upsample(self.size,self.scale_factor,mode=self.mode)
   

    def forward(self,input):
        # self.cpu()
        temp=torch.zeros(time_window,input.size()[1],input.size()[2],input.size()[3]*self.scale_factor,input.size()[4]*self.scale_factor, device=input.device)
        # print(temp.device,'-----')
        for i in range(time_window):
            
            temp[i]=self.up(input[i])

            # temp[i]= F.interpolate(input[i], scale_factor=self.scale_factor,mode='nearest')
        return temp



class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel=3, stride=1,e=0.5):
        super().__init__()
        c_ = int(out_channels * e) 
        
        self.cv1 = Conv(in_channels, c_, k=kernel, s=stride)
        self.cv2 = Conv(c_, out_channels, 3, 1)
        # self.shortcut=Conv_2(in_channels,out_channels,k=1,s=stride)
        self.shortcut = nn.Sequential(
            )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                batch_norm_2d(out_channels),
            )

    def forward(self, x):   
        return (self.cv2(self.cv1(x)) + self.shortcut(x))
    
class BasicBlock_1(nn.Module):#
    def __init__(self, in_channels, out_channels, stride=1,e=0.5):
        super().__init__()
        # c_ = int(out_channels * e)  # hidden channels  
        c_=1024
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, c_, kernel_size=3, stride=stride, padding=1, bias=False),
            batch_norm_2d(c_),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=3, padding=1, bias=False),
            batch_norm_2d1(out_channels),
            )
        # shortcut
        self.shortcut = nn.Sequential(
            )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )
   
            
    def forward(self, x):
        # print(self.residual_function(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class BasicBlock_2(nn.Module):
    def __init__(self, in_channels, out_channels,k_size=3,stride=1):
        super().__init__()
        p=None
        if k_size==3:
            pad=1
        if k_size==1:
            pad=0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
            )
        self.shortcut = nn.Sequential(
            )
      
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),
                mem_update(act=False),
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )
            
    def forward(self, x):
        return (self.residual_function(x) + self.shortcut(x))


class Concat_res2(nn.Module):#
    def __init__(self, in_channels, out_channels,k_size=3, stride=1,e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels  
        if k_size==3:
            pad=1
        if k_size==1:
            pad=0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(out_channels),
            mem_update(act=False),
            Snn_Conv2d(out_channels, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
            )
        # shortcut
        self.shortcut = nn.Sequential(
            )
            
        if in_channels<out_channels:
            self.shortcut = nn.Sequential(                 
                mem_update(act=False),       
                Snn_Conv2d(in_channels, out_channels-in_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels-in_channels),
            )
        self.pools=nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        # print(self.residual_function(x).shape)
        temp=self.shortcut(x)
        out=torch.cat((temp,x),dim=2)
        out=self.pools(out)
        return (self.residual_function(x) + out)


class BasicBlock_ms(nn.Module):#tiny3.yaml
    def __init__(self, in_channels, out_channels,k_size=3, stride=1,e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels  
        if k_size==3:
            pad=1
        if k_size==1:
            pad=0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, c_, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(c_),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
            )
        # shortcut
        self.shortcut = nn.Sequential(
            )
        if stride != 1 or in_channels != out_channels:
        
            self.shortcut = nn.Sequential(
                nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride)),    
                Snn_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels),
            )
        
    def forward(self, x):
        # print(self.residual_function(x).shape)
        return (self.residual_function(x) + self.shortcut(x))


class ConcatBlock_ms(nn.Module):#
    def __init__(self, in_channels, out_channels,k_size=3, stride=1,e=0.5):
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels  
        if k_size==3:
            pad=1
        if k_size==1:
            pad=0
        self.residual_function = nn.Sequential(
            mem_update(act=False),
            Snn_Conv2d(in_channels, c_, kernel_size=k_size, stride=stride, padding=pad, bias=False),
            batch_norm_2d(c_),
            mem_update(act=False),
            Snn_Conv2d(c_, out_channels, kernel_size=k_size, padding=pad, bias=False),
            batch_norm_2d1(out_channels),
            )
        # shortcut
        self.shortcut = nn.Sequential(
            )
            
        if in_channels<out_channels:
            self.shortcut = nn.Sequential(                 
                mem_update(act=False),       
                Snn_Conv2d(in_channels, out_channels-in_channels, kernel_size=1, stride=1, bias=False),
                batch_norm_2d(out_channels-in_channels),
            )
        self.pools=nn.MaxPool3d((1, stride, stride), stride=(1, stride, stride))

    def forward(self, x):
        # print(self.residual_function(x).shape)
        temp=self.shortcut(x)
        out=torch.cat((temp,x),dim=2)
        out=self.pools(out)
        return (self.residual_function(x) + out)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)



