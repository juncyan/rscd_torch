import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import cv2
from PIL import Image
import os
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import torch.functional as F
import numpy as np
import requests

from f3net.f3net import F3Net

from dataset.CDReader import CDReader, TestReader


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

dataset_name = "LEVIR_CD"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

wp = r"/home/jq/Code/torch/output/levir_cd/F3Net_2024_04_12_20/iter_100.pth"
layer_state_dict = torch.load(f"{wp}")

model = F3Net()
model_name = "F3Net"#model.__str__().split("(")[0]
target_layers = [model.classier]

model.load_state_dict(layer_state_dict)
model = model.eval()
model = model.cuda()
test_data = TestReader(dataset_path, mode="train",en_edge=False)
loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0,shuffle=True, drop_last=True)

save_dir = f"cam/{dataset_name}/{model_name}"
if not os.path.exists(save_dir):
     os.makedirs(save_dir)

color_label = np.array([[0,0,0],[255,255,255]])
for _, (im1, im2, l1, na) in enumerate(loader):
    
    tensor = torch.cat([im1, im2], 1).cuda()
    img1 = im1.squeeze().cpu().numpy()
    img1 = np.transpose(img1, [1,2,0])
    img2 = im2.squeeze().cpu().numpy()
    img2 = np.transpose(img2, [1,2,0])
    label = l1.numpy()
    lm = np.argmax(label, 1)
    lm = lm.squeeze()
    lm = color_label[lm] / 255.0
    name = na[0]

    targets = [SemanticSegmentationTarget(1, label)]


    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

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

    
    

