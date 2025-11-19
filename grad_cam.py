import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import cv2
from PIL import Image
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import torch.functional as F
import numpy as np
import requests
import argparse

from cd_models.mambacd import build_STMambaBCD
from cd_models.eafhnet import EAFHNet
from cd_models.lkmamba_cd import LKMamba_CD
from cd_models.isdanet import ISDANet
from cd_models.lwganet.lwclafr import CLAFR_LWGA
from core.datasets import SBCDReader, CDReader
from cd_models.mfnet import MFNet


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


def normalize(img, mean=[0.485, 0.456, 0.406], std=[1, 1, 1]):
        # return img
        im = img.astype(np.float32, copy=False)
        mean = np.array(mean)[np.newaxis, np.newaxis, :]
        std = np.array(std)[np.newaxis, np.newaxis, :]
        im = im / 255.0
        im -= mean
        im /= std
        return im

def parse_args():
    parser = argparse.ArgumentParser(description='Semantic Segmentation Overfitting Test')
    # model
    parser.add_argument('--model', type=str, default='fssh',
                        help='model name (default: msfgnet)')
    parser.add_argument('--root', type=str, default='./output',
                        help='run save dir (default: ./output)')
    parser.add_argument('--img_size', type=int, default=256,
                        help='input image size (default: 256)')
    parser.add_argument('--device', type=int, default=1,
                        choices=[-1, 0, 1],
                        help='device (default: gpu:0)')
    parser.add_argument('--dataset', type=str, default="MacaoCD",
                        help='dataset name (default: MacaoCD)')
    parser.add_argument('--iters', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--en_load_edge', type=bool, default=False,
                        help='en_load_edge False')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='num classes (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size (default: 8)')
    parser.add_argument('--lr', type=float, default=2.8e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num_workers (default: 16)')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    args = parser.parse_args()
    return args
args = parse_args()
dataset_name = "MacaoCD"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

wp = r"/home/jq/Code/torch/output/macaocd/ISDANet_2025_03_08_14/ISDANet_best.pth"
layer_state_dict = torch.load(f"{wp}")

model = ISDANet()
model_name = "ISDANet_test2"#model.__str__().split("(")[0]
target_layers = [model.up1]  # for MambaBCD
print(model_name)
model.load_state_dict(layer_state_dict)
model = model.eval()
model = model.cuda()
test_data = CDReader(dataset_path, mode="test")
loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0,shuffle=True, drop_last=True)

save_dir = f"/mnt/data/Results/cam/{dataset_name}/{model_name}"
if not os.path.exists(save_dir):
     os.makedirs(save_dir)

color_label = np.array([[0,0,0],[255,255,255]])
for im1, im2, l1, na in tqdm(loader):
    
    tensor = torch.cat([im1, im2], 1).cuda()
    img1 = im1.squeeze().cpu().numpy()
    img1 = np.transpose(img1, [1,2,0])
    img2 = im2.squeeze().cpu().numpy()
    img2 = np.transpose(img2, [1,2,0])
    label = l1.numpy()
    lml = np.argmax(label, 1, keepdims=True)
    lm = lml.squeeze()
    lm = color_label[lm] / 255.0
    name = na[0]
    
    targets = [SemanticSegmentationTarget(0, lml)]

    img1 = normalize(img1)
    img2 = normalize(img2)

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    # grayscale_cam = np.array(grayscale_cam == 0, dtype=np.uint8)
    grayscale_cam = grayscale_cam[0, :, :]

    visualization = show_cam_on_image(img1, grayscale_cam, use_rgb=True)
    img = Image.fromarray(visualization)
    img.save(os.path.join(save_dir, name.replace(".png","_A.png")))
    # cv2.imwrite(os.path.join(save_dir, name.replace(".png","_A.png")), visualization)

    visualization = show_cam_on_image(img2, grayscale_cam, use_rgb=True)
    img = Image.fromarray(visualization)
    img.save(os.path.join(save_dir, name.replace(".png","_B.png")))
    # cv2.imwrite(os.path.join(save_dir, name.replace(".png","_B.png")), visualization)

    visualization = show_cam_on_image(lm, grayscale_cam, use_rgb=True)
    img = Image.fromarray(visualization)
    img.save(os.path.join(save_dir, name.replace(".png","_L.png")))
    # cv2.imwrite(os.path.join(save_dir, name.replace(".png","_L.png")), visualization)

    
    