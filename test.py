# 调用官方库及第三方库
import torch
import numpy as np
import argparse
import datetime
import platform
import random
import os

# 模型导入

from cd_models.mscanet.model import MSCANet
from cd_models.aernet import AERNet
from cd_models.ResUnet import ResUnet
from cd_models.a2net import LightweightRSCDNet
from cd_models.ussfcnet.ussfcnet import USSFCNet
from cd_models.dtcdscn import DTCDSCNet
from cd_models.changeformer import ChangeFormerV6
from cd_models.dminet import DMINet
from cd_models.siamunet_diff import SiamUnet_diff
from cd_models.siamunet import SiamUnet_conc
from cd_models.SNUNet import SNUNet
from cd_models.dsamnet import DSAMNet
from cd_models.stanet import STANetSA
from cd_models.icifnet import ICIFNet
from cd_models.dsifn import DSIFN
from cd_models.bit_cd import BIT_CD
from cd_models.transunet import TransUNet
from cd_models.rdpnet import RDPNet
from cd_models.bisrnet import BiSRNet, SSCDl
from cd_models.hanet import HAN
from cd_models.cgnet import CGNet
# from cd_models.rsmamba import RSMamba_CD
# from cd_models.mambacd import build_STMambaSCD
# from cd_models.scd_sam import SCD_SAM
from cd_models.dgma2net import DGMAANet
from models.model_cd import RLM_CD_v2
from models.lkssm_cd_v0 import RepLKSSM_CD_v0
from cd_models.vmamba import build_VSSM, Backbone_VSSM
from core.bcdwork import Work

# dataset_name = "GVLM_CD"
# dataset_name = "LEVIR_CD"
# dataset_name = "CLCD"
# dataset_name = "SYSU_CD"

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='fssh',
                        help='model name (default: msfgnet)')
    parser.add_argument('--root', type=str, default='./output',
                        help='model name (default: ./output)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input image size (default: 256)')
    parser.add_argument('--device', type=int, default=0,
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default="CLCD",
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--img_ab_concat', type=bool, default=True,
                        help='img_ab_concat False')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num classes (default: 2)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch_size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 2.8e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num_workers (default: 8)')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    print("main")
    # args = parse_args()
    x = torch.rand([1,3,244,244]).cuda()
    m = build_VSSM("base").cuda()
    y = m(x)
    
    
    