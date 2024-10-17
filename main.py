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
from cd_models.rsmamba import RSMamba_CD
from cd_models.mambacd import build_STMambaBCD

from core.scdwork import Work

# dataset_name = "GVLM_CD"
# dataset_name = "LEVIR_CD"
# dataset_name = "CLCD"
# dataset_name = "SYSU_CD"

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='fssh',
                        help='model name (default: msfgnet)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input image size (default: 256)')
    parser.add_argument('--device', type=int, default=0,
                        choices=[0, 1],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default="MacaoCD",
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=7,
                        help='num classes (default: 7)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num_workers (default: 8)')
    parser.add_argument('--cfg', type=str, default='/home/jq/Code/torch/cd_models/mambacd/configs/vssm1/vssm_base_224.yaml',
                        help='train mamba')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 代码运行预处理
    print("main")
    args = parse_args()
    # model = build_STMambaBCD(args)
    model = SSCDl(in_channels=3, num_classes=args.num_classes)
    w = Work(model, args,'./output')
    w()
    
    
    