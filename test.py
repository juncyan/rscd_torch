import numpy as np
import torch
import argparse
from imageio.v3 import imread

from demo import Demo_CD, Demo_SCD
from demo import bcd_inference, scd_inference

def parse_args():
    parser = argparse.ArgumentParser(description='Remote Sensing Change Detection Demo')
    # model
    parser.add_argument('--task', type=str, default=r'scd', choices=[r'bcd', r'scd'])
    parser.add_argument('--save_path', type=str, default=r'./output',
                        help='run save dir (default: ./output)')
    parser.add_argument('--img1', type=str, default=r"./images/scd01A.png")
    parser.add_argument('--img2', type=str, default=r"./images/scd01B.png")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args = parser.parse_args()
    return args



def main(args):

    img1 = imread(args.img1)[np.newaxis, :,:,:3]
    img2 = imread(args.img2)[np.newaxis, :,:,:3]
    imgs = np.concatenate([img1, img2], axis=-1)
    imgs = torch.from_numpy(imgs).permute(0,3,1,2).float().cuda()

    if args.task == "scd":
        model = Demo_SCD(256, 5).cuda()
        weight_path = r".demo/Demo_SCD_best.pth"
        scd_inference(model, imgs, weight_path=weight_path, save_path=args.save_path)
    else:
        weight_path = r".demo/Demo_CD_best.pth"
        model = Demo_CD(256).cuda()
        bcd_inference(model, imgs, weight_path=weight_path, save_path=args.save_path)

if __name__ == "__main__":
    # 代码运行预处理
    print("main")
    args = parse_args()
    main(args)