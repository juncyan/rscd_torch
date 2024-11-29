import torch
import torch.nn as nn
from typing import Optional, Callable, Any
import torch.utils.checkpoint as checkpoint

from cd_models.unireplknet import get_conv2d, get_bn, fuse_bn, merge_dilated_into_large_kernel, NCHWtoNHWC,NHWCtoNCHW, GRNwithNHWC

from cd_models.vmamba import *


class LKSSMBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)

        else:
            assert kernel_size in [3, 5]
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4) #ChannelOperation(dim)#

        # self.ss2d = SS2D_v3(dim, dim)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        

    def compute_residual(self, x):
        #model 3
        # y = self.se(self.norm(self.dwconv(x)))
        # y2 = self.se(x)
        # y = y1 + y2
        # y = self.pwconv2(self.act(self.pwconv1(y)))
        #model 2
        # y1 = self.se(self.norm(self.dwconv(x)))
        # y2 = self.pwconv2(self.act(self.pwconv1(x)))
        # y = y1 + y2
        # model 1
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)
    
    # def ssm(self, x):
    #     y = x + self.ss2d(x)
    #     return y

    def forward(self, inputs):

        def _f(x):
            # return self.compute_residual(x)  + self.ssm(x)
            return x + self.compute_residual(x) 

        if self.with_cp and inputs.requires_grad:
            out = checkpoint.checkpoint(_f, inputs)
        else:
            out = _f(inputs)
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                            self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 padding=self.dwconv.padding, groups=self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)

class SS2D_v3(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        in_ch=96,
        out_ch=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # dwconv ===============
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        channel_first=True,
        # ======================
        **kwargs,
    ):
        super().__init__()
        d_model = in_ch
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.channel_first = channel_first
    

        self.forward_core = partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex)
        k_group = 4

        self.conv3 = nn.Conv2d(in_ch, d_inner, 1)
        self.conv4 = nn.Conv2d(d_inner, out_ch, 1)

        if channel_first:
            self.out_norm_shape = "v1"
            self.out_norm = LayerNorm2d(d_inner)
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner)
        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
        
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    def forward_corev2(self, x: torch.Tensor, cross_selective_scan=cross_selective_scan, **kwargs):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")

        return cross_selective_scan(
            x, x_proj_weight, None, dt_projs_weight, dt_projs_bias,
            A_logs, Ds, delta_softplus=True,
            out_norm=out_norm,
            channel_first=self.channel_first,
            out_norm_shape=out_norm_shape,
            **kwargs,
        )

    def forward(self, x: torch.Tensor):
        y = self.conv3(x)
        z = self.forward_core(y)
        out = z + y
        out = self.conv4(out)
        return out


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy=False, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3]
            self.dilates = [1, 2, 4, 5]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3]
            self.dilates = [1, 2, 3, 5]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3]
            self.dilates = [1, 2, 4]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)
