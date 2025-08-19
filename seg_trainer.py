import argparse

# 模型导入
# from cd_models.ultralight_unet import UltraLightUNet
# from cd_models.unet import net_factory

from models.fgfp import FGFPVM_Seg
from core.segwork import Work

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
    parser.add_argument('--img_size', type=int, default=512,
                        help='input image size (default: 256)')
    parser.add_argument('--device', type=int, default=0,
                        choices=[-1, 0, 1],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default="Landcover",
                        help='dataset name (default: LEVIR_CD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='num classes (default: 7)')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch_size (default: 2)')
    parser.add_argument('--lr', type=float, default=0.00035, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num_workers (default: 16)')
    # parser.add_argument('--cfg', type=str, default='/home/jq/Code/torch/cd_models/mambacd/configs/vssm1/vssm_tiny_224_0229flex.yaml',
    #                     help='train mamba')
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
    # model = UltraLightUNet(num_classes=args.num_classes)
    # model = RepLKSSM_Seg(args.num_classes)
    model = FGFPVM_Seg(args.img_size, args.num_classes)
    w = Work(model, args)
    
    
    
    