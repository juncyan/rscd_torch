import torch
from torch import nn
from einops import rearrange
from cd_models.utils.mmcv_load_checkpoint import _load_checkpoint
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from cd_models.vmamba.vmamba import SS2D, PatchMerging2D, DropPath, LayerNorm2d, Mlp
from .lkm import LKSSMBlock

class Mamba_LK(nn.Module):
    def __init__(self):
        super().__init__()
        self.vss1 = LKSSMBlock(128, 13)
        self.dconv1 = nn.Sequential(nn.Conv2d(128, 160, 3, 2, 1), nn.BatchNorm2d(160), nn.ReLU())
        self.vss2 = LKSSMBlock(160)
        self.dconv2 = nn.Sequential(nn.Conv2d(160, 320, 3, 2, 1), nn.BatchNorm2d(320), nn.ReLU())
        self.vss3 = LKSSMBlock(320)
        self.dconv3 = nn.Sequential(nn.Conv2d(320, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, b1, b2, b3, b):
        b1 = b1.contiguous()
        b2 = b2.contiguous()
        b3 = b3.contiguous()
        b = b.contiguous()
        t1 = self.vss1(b1)
        # t1 = rearrange(t1, 'b h w c-> b c h w')
        t = self.dconv1(t1)
        # t = rearrange(t, 'b c h w-> b h w c')
        t2 = t + b2
        t2 = self.vss2(t2)
        # t2 = rearrange(t2, 'b h w c-> b c h w')
        t = self.dconv2(t2)
        # t = rearrange(t, 'b c h w-> b h w c')
        t3 = t + b3
        t3 = self.vss3(t3)
        # t3 = rearrange(t3, 'b h w c-> b c h w')
        t = self.dconv3(t3)
        t4 = t + b
        return t1, t2, t3, t4
    
    def init_weights(self):

        def load_state_dict(module, state_dict, strict=False, logger=None):
            unexpected_keys = []
            own_state = module.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    unexpected_keys.append(name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            missing_keys = set(own_state.keys()) - set(state_dict.keys())

            err_msg = []
            if unexpected_keys:
                err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
            if missing_keys:
                err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
            err_msg = '\n'.join(err_msg)
            if err_msg:
                if strict:
                    raise RuntimeError(err_msg)
                elif logger is not None:
                    logger.warn(err_msg)
                else:
                    print(err_msg)

        logger = None #get_root_logger()
        assert self.init_cfg is not None
        ckpt_path = self.init_cfg['checkpoint']
        if ckpt_path is None:
            print('================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization')
        else:
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            load_state_dict(self, _state_dict, strict=False, logger=logger)


    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

class Mamba_Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.vss1 = VSSBlock(128)
        self.dconv1 = nn.Sequential(nn.Conv2d(128, 160, 3, 2, 1), nn.BatchNorm2d(160), nn.ReLU())
        self.vss2 = VSSBlock(160)
        self.dconv2 = nn.Sequential(nn.Conv2d(160, 320, 3, 2, 1), nn.BatchNorm2d(320), nn.ReLU())
        self.vss3 = VSSBlock(320)
        self.dconv3 = nn.Sequential(nn.Conv2d(320, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, b1, b2, b3, b):
        t1 = self.vss1(b1)
        t1 = rearrange(t1, 'b h w c-> b c h w')
        t = self.dconv1(t1)
        t = rearrange(t, 'b c h w-> b h w c')
        t2 = t + b2
        t2 = self.vss2(t2)
        t2 = rearrange(t2, 'b h w c-> b c h w')
        t = self.dconv2(t2)
        t = rearrange(t, 'b c h w-> b h w c')
        t3 = t + b3
        t3 = self.vss3(t3)
        t3 = rearrange(t3, 'b h w c-> b c h w')
        t = self.dconv3(t3)
        t4 = t + b
        return t1, t2, t3, t4



class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: nn.Module = nn.LayerNorm,
        channel_first=False,
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        gmlp=False,
        # =============================
        use_checkpoint: bool = False,
        post_norm: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                channel_first=channel_first,
            )
        
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=channel_first)

    def forward(self, input: torch.Tensor):
        if self.ssm_branch:
            if self.post_norm:
                x = input + self.drop_path(self.norm(self.op(input)))
            else:
                x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

