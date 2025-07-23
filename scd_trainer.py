# 调用官方库及第三方库
import torch
import numpy as np
import argparse
import datetime
import platform
import random
import os

# 模型导入

from cd_models.mambacd import build_STMambaSCD
from cd_models.scd_sam import SCD_SAM
from cd_models.bisrnet import BiSRNet, SSCDl
from cd_models.daudt.HRSCD3 import HRSCD3
from cd_models.daudt.HRSCD4 import HRSCD4
from cd_models.ssesn import SSESN
from cd_models.cdsc import CDSC
from cd_models.btscd import BTSCD
from cd_models.cienet import FESCD_VMB
# from scdmodel.scd import SCDSam_CrossA
# from scdmodel.sam_mamba import SCDSamMamba, SCDSamMambaLK

from core.scdwork import Work
# from core.secondwork import Work

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
    parser.add_argument('--device', type=int, default=1,
                        choices=[-1, 0, 1],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default="MusSCD",
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='num classes (default: 7)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch_size (default: 2)')
    parser.add_argument('--lr', type=float, default=0.00035, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num_workers (default: 16)')
    # parser.add_argument('--cfg', type=str, default='/home/jq/Code/torch/cd_models/vmamba/configs/vssm_base_224.yaml',
    #                     help='train mamba')
    # parser.add_argument('--pretrained_weight_path', type=str, default="/home/jq/Code/weights/vssm_tiny_0230_ckpt_epoch_262.pth")
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
    # model = build_STMambaSCD(args)
    # model = SSCDl(in_channels=3, num_classes=args.num_classes)
    # model = SCD_SAM(input_size=args.img_size, num_classes=args.num_classes)
    # model = CDSC(output_nc=args.num_classes)
    # model = SSESN(n_classes=args.num_classes)
    # model = HRSCD4(3, args.num_classes)
    # model = BiSRNet(num_classes=args.num_classes)
    # model = SCDSamMambaLK(img_size=args.img_size, num_seg=args.num_classes)
    # model = FESCD_VMB(args.img_size, args.num_classes)
    model = BTSCD(num_classes=args.num_classes)
    w = Work(model, args)
    
    
    
    