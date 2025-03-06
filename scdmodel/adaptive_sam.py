import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath as TimmDropPath,\
    to_2tuple, trunc_normal_
from timm.models.registry import register_model
from typing import Tuple
from collections import OrderedDict
from typing import Dict, List, Tuple, Union
import numpy as np
from cd_models.mobilesam import Sam, TinyViT
from .lkm import LKSSMBlock, MambaLayer

class MSAMMamba(nn.Module):
    def __init__(self, img_size=256):
        super().__init__()
        self.sam = Sam(
            image_encoder=TinyViT(img_size=img_size, in_chans=3, num_classes=1000,
                embed_dims=[64, 128, 160, 320],
                depths=[2, 2, 6, 2],
                num_heads=[2, 4, 5, 10],
                window_sizes=[7, 7, 14, 7],
                mlp_ratio=4.,
                drop_rate=0.,
                drop_path_rate=0.0,
                use_checkpoint=False,
                mbconv_expand_ratio=4.0,
                local_conv_size=3,
                layer_lr_decay=0.8))
    
        model_dict = self.sam.state_dict()
        sam_checkpoint = "/home/jq/Code/weights/mobile_sam.pt"
        pretrained_dict = torch.load(sam_checkpoint)

        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                            if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.sam.load_state_dict(OrderedDict(model_dict), strict=True)
        self.sam.eval()
        for param in self.sam.parameters():
            param.requires_grad = False
        
        self.mamba = LKSSMBlock(64, 13)
    
    def forward(self, inputs):
        x = self.sam.image_encoder.patch_embed(inputs)
        x = self.mamba(x)
        x = self.sam.image_encoder.layers[0](x)
        start_i = 1
        features=[x]
        B,N,C=x.size()
        WH=int(np.sqrt(N))
        x1 = x.view(B, WH, WH, C)
        x1=x1.permute(0, 3, 1, 2)
        # features.append(x1)
        for i in range(start_i, len(self.sam.image_encoder.layers)):
            layer = self.sam.image_encoder.layers[i]
            x = layer(x)
            # x2=self.convert(x)
            features.append(x)
     
        B,N,C=x.size()
                                    
        WH = int(np.sqrt(N))
        x = x.view(B, WH, WH, C)
        x = x.permute(0, 3, 1, 2)
        x = self.sam.image_encoder.neck(x)
        features.append(x)
        return features
