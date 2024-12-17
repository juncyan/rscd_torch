# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import argparse
import random
import os

# 基础功能

from cd_models.mambacd import build_STMambaSCD
from cd_models.scd_sam import SCD_SAM
from cd_models.bisrnet import BiSRNet, SSCDl
from cd_models.daudt.HRSCD3 import HRSCD3
from cd_models.daudt.HRSCD4 import HRSCD4
from cd_models.ssesn import SSESN
from cd_models.cdsc import CDSC
from core.scdmisc.predict import predict

from core.datasets.SCDReader import MusReader, SCDReader

# class parameter:
#     lr = params["lr"]
#     momentum = params["momentum"]
#     weight_decay = params["weight_decay"]
#     num_epochs = num_epochs
#     batch_size = batch_size

# dataset_name = "GVLM_CD_d"
# dataset_name = "LEVIR_c"
# dataset_name = "CLCD"
# dataset_name = "SYSCD_d"
dataset_name = "MusSCD"
dataset_name = 'Second'

dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='fssh',
                        help='model name (default: msfgnet)')
    parser.add_argument('--root', type=str, default='./output',
                        help='model name (default: ./output)')
    parser.add_argument('--img_size', type=int, default=512,
                        help='input image size (default: 256)')
    parser.add_argument('--device', type=int, default=0,
                        choices=[-1, 0, 1],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default="Second",
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=7,
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
    parser.add_argument('--cfg', type=str, default='/home/jq/Code/VMamba/changedetection/configs/vssm1/vssm_base_224.yaml',
                        help='train mamba')
    parser.add_argument('--pretrained_weight_path', type=str, default="/home/jq/Code/weights/vssm_tiny_0230_ckpt_epoch_262.pth")
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
    torch.cuda.empty_cache()
    torch.cuda.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    args = parse_args()
    test_data = SCDReader(dataset_path, "test")#MusReader(dataset_path, mode="val")
    
    weight_path = r"/home/jq/Code/torch/output/second/CDSC_2024_12_10_01/CDSC_best.pth"
    model = CDSC(output_nc=7)

    predict(model, test_data, weight_path, dataset_name,7,1)
    # x = torch.rand([1,3,256,256]).cuda()
    # y = model(x,x)
    # print(y.shape)    
