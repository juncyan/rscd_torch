import os
import re
import torch
import torch.nn as nn
from collections import OrderedDict
from .models.block.Base import ChannelChecker
from .models.Encoder.DFI import *
from .models.block.Base import Conv1Relu
from .models.Encoder.mobilesam import build_sam_vit_t
from .models.Encoder.moat import *
from .models.Decoder.FPN import FPNNeck
from .models.Decoder.AFPN_CARAFE import AFPN
from .models.head.FCN import FCNHead, CSD, FCNHeadCD

# @ARTICLE{10543161,
#   author={Mei, Liye and Ye, Zhaoyi and Xu, Chuan and Wang, Hongzhu and Wang, Ying and Lei, Cheng and Yang, Wei and Li, Yansheng},
#   journal={IEEE Transactions on Geoscience and Remote Sensing}, 
#   title={SCD-SAM: Adapting Segment Anything Model for Semantic Change Detection in Remote Sensing Imagery}, 
#   year={2024},
#   volume={62},
#   number={},
#   pages={1-13},
#   keywords={Semantics;Feature extraction;Remote sensing;Image segmentation;Decoding;Adaptation models;Transformers;Progressive feature aggregation;segment anything model (SAM);semantic adaptor;semantic change detection (SCD)},
#   doi={10.1109/TGRS.2024.3407884}}

class SCD_SAM(nn.Module):
    def __init__(self, input_size=512, num_classes=7, pretrain=None):
        super().__init__()
        self.inplanes = 96 
        num_classes = num_classes
        feat_num = 16 if input_size==512 else int(np.sqrt(input_size) // 2)

        self.SAM_Encoder = build_sam_vit_t(img_size=input_size)
        self.CNN_Encoder = moat_4(use_window=True, num_classes=10)
        self.Binary_Decoder = FPNNeck(self.inplanes)
        self.Semantic_Decoder = AFPN(self.inplanes)

        self.CSD = CSD(in_dim=self.inplanes, num_classes=self.inplanes)  
        self.head = FCNHead(self.inplanes, num_classes, 1)

        if not pretrain is None:
            self._init_weight(pretrain)   
        self.check_channels = ChannelChecker(self.SAM_Encoder, self.inplanes, input_size)
        self.fusion4 = DFI(self.inplanes*8, self.inplanes*8, feat_num, feat_num)
        self.conv4 = Conv1Relu(self.inplanes*16, self.inplanes*8)

        self.SAM_Encoder.eval()
        for param in self.SAM_Encoder.parameters():
            param.requires_grad = False


    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, _,fa4 = self.SAM_Encoder(xa)  
        fa1, fa2, fa3,fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, _,fb4 = self.SAM_Encoder(xb)
        fb1, fb2, fb3,fb4 = self.check_channels(fb1, fb2, fb3, fb4)
        fa12, fa22, fa32, fa42 = self.CNN_Encoder(xa)
        fb12, fb22, fb32, fb42 = self.CNN_Encoder(xb)

        inentity_a4 = fa4
        inentity_b4 = fb4
        fa4 = self.conv4(torch.cat([inentity_a4, self.fusion4(fa4, fa42)], 1))
        fb4 = self.conv4(torch.cat([inentity_b4, self.fusion4(fb4, fb42)], 1))

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4   

        change = self.Binary_Decoder(ms_feats)
        change_s1 = self.Semantic_Decoder(ms_feats)
        change_s2 = self.Semantic_Decoder(ms_feats)

        change_s1, change_s2 = self.CSD(change_s1, change_s2)
        out_s1, out_s2, out = self.head(change_s1, change_s2, change, out_size=(h_input, w_input))

        return out, out_s1, out_s2


    def _init_weight(self, pretrain=''):  
        for m in self.modules():
            if isinstance(m, nn.Conv2d): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):  
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if pretrain.endswith('.pt'):
            pretrained_dict = torch.load(pretrain)
            if isinstance(pretrained_dict, nn.DataParallel):
                pretrained_dict = pretrained_dict.module
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.state_dict().items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(OrderedDict(model_dict), strict=True)
            print("=> ChangeDetection load {}/{} items from: {}".format(len(pretrained_dict),
                                                                        len(model_dict), pretrain))


class SCD_SAM_BCD(nn.Module):
    def __init__(self, input_size=512, num_classes=7, pretrain=None):
        super().__init__()
        self.inplanes = 96 
        num_classes = num_classes
        feat_num = 16 if input_size==512 else int(np.sqrt(input_size) // 2)

        self.SAM_Encoder = build_sam_vit_t(img_size=input_size)
        self.CNN_Encoder = moat_4(use_window=True, num_classes=10)
        self.Binary_Decoder = FPNNeck(self.inplanes)
        # self.Semantic_Decoder = AFPN(self.inplanes)

        # self.CSD = CSD(in_dim=self.inplanes, num_classes=self.inplanes)  
        self.head = FCNHeadCD(self.inplanes, num_classes, 2)

        self.check_channels = ChannelChecker(self.SAM_Encoder, self.inplanes, input_size)
        self.fusion4 = DFI(self.inplanes*8, self.inplanes*8, feat_num, feat_num)
        self.conv4 = Conv1Relu(self.inplanes*16, self.inplanes*8)
        self.SAM_Encoder.eval()
        for param in self.SAM_Encoder.parameters():
            param.requires_grad = False


    def forward(self, xa, xb):
        _, _, h_input, w_input = xa.shape
        assert xa.shape == xb.shape, "The two images are not the same size, please check it."

        fa1, fa2, fa3, _,fa4 = self.SAM_Encoder(xa)  
        fa1, fa2, fa3,fa4 = self.check_channels(fa1, fa2, fa3, fa4)
        fb1, fb2, fb3, _,fb4 = self.SAM_Encoder(xb)
        fb1, fb2, fb3,fb4 = self.check_channels(fb1, fb2, fb3, fb4)
        fa12, fa22, fa32, fa42 = self.CNN_Encoder(xa)
        fb12, fb22, fb32, fb42 = self.CNN_Encoder(xb)

        inentity_a4 = fa4
        inentity_b4 = fb4
        fa4 = self.conv4(torch.cat([inentity_a4, self.fusion4(fa4, fa42)], 1))
        fb4 = self.conv4(torch.cat([inentity_b4, self.fusion4(fb4, fb42)], 1))

        ms_feats = fa1, fa2, fa3, fa4, fb1, fb2, fb3, fb4   

        change = self.Binary_Decoder(ms_feats)
        # change_s1 = self.Semantic_Decoder(ms_feats)
        # change_s2 = self.Semantic_Decoder(ms_feats)

        # change_s1, change_s2 = self.CSD(change_s1, change_s2)
        out = self.head(change, out_size=(h_input, w_input))

        return out

