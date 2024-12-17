import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from cd_models.ultralight_unet import act_layer, _init_weights
from timm.models.helpers import named_apply
from functools import partial

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    # from mmseg.utils import get_root_logger
    from cd_models.utils.mmcv_load_checkpoint import _load_checkpoint
    has_mmseg = True
except ImportError:
    _load_checkpoint = None


from .replk import LKSSMBlock

depths=(3, 3, 27, 3)
dims=(96, 192, 384, 768)
cfg = {"depths":[2,2,4,2],
       "dims":[96, 192, 384, 768],
       "kernels":[11,13,13,11],
       "down_ratio":[4,2,2,2]}


class LKSSMNet(nn.Module):
    def __init__(self, cfg=cfg) -> None:
        super().__init__()
        depths = cfg['depths']
        dims = cfg['dims']
        kernels = cfg['kernels']
        down_ratio = cfg['down_ratio']

        in_ch = 3
        self.opt = nn.ModuleList()
        for i, num_layer in enumerate(depths):
            downsample = nn.Sequential(nn.Conv2d(in_ch, dims[i], 1),nn.BatchNorm2d(dims[i]), nn.ReLU6(),
                                       nn.Conv2d(dims[i], dims[i], down_ratio[i] + 1, stride=down_ratio[i], padding=down_ratio[i]//2, groups=dims[i]), 
                                       nn.BatchNorm2d(dims[i]), nn.ReLU())
            in_ch = dims[i]
            sub_layer = nn.Sequential()
            sub_layer.append(downsample)
            for _ in range(num_layer):
                sub_layer.append(LKSSMBlock(dims[i], kernels[i]))
            
            self.opt.append(sub_layer)
        
        self.init_weights('normal')
    
    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)
        
    def forward(self, x):
        res = []
        y = x
        for lay in self.opt:
            y = lay(y)
            res.append(y)
        return res
    
    # def init_weights(self):

    #     def load_state_dict(module, state_dict, strict=False, logger=None):
    #         unexpected_keys = []
    #         own_state = module.state_dict()
    #         for name, param in state_dict.items():
    #             if name not in own_state:
    #                 unexpected_keys.append(name)
    #                 continue
    #             if isinstance(param, torch.nn.Parameter):
    #                 # backwards compatibility for serialized parameters
    #                 param = param.data
    #             try:
    #                 own_state[name].copy_(param)
    #             except Exception:
    #                 raise RuntimeError(
    #                     'While copying the parameter named {}, '
    #                     'whose dimensions in the model are {} and '
    #                     'whose dimensions in the checkpoint are {}.'.format(
    #                         name, own_state[name].size(), param.size()))
    #         missing_keys = set(own_state.keys()) - set(state_dict.keys())

    #         err_msg = []
    #         if unexpected_keys:
    #             err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
    #         if missing_keys:
    #             err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
    #         err_msg = '\n'.join(err_msg)
    #         if err_msg:
    #             if strict:
    #                 raise RuntimeError(err_msg)
    #             elif logger is not None:
    #                 logger.warn(err_msg)
    #             else:
    #                 print(err_msg)

        
    #     assert self.init_cfg is not None
    #     ckpt_path = self.init_cfg['checkpoint']
    #     if ckpt_path is None:
    #         print('================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization')
    #     else:
    #         ckpt = _load_checkpoint(ckpt_path, logger=None, map_location='cpu')
    #         if 'state_dict' in ckpt:
    #             _state_dict = ckpt['state_dict']
    #         elif 'model' in ckpt:
    #             _state_dict = ckpt['model']
    #         else:
    #             _state_dict = ckpt

    #         load_state_dict(self, _state_dict, strict=False, logger=None)


    # def _init_weights(self, m):
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         trunc_normal_(m.weight, std=.02)
    #         if hasattr(m, 'bias') and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)

    def reparameterize_unireplknet(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()
                