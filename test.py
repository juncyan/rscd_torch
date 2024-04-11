import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import cv2
import os
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import torch.functional as F
import numpy as np
import requests
from PIL import Image

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

dataset_name = "CLCD"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)

wp = r"output/clcd/F3Net_2024_04_11_14/F3Net_best.pth"
layer_state_dict = torch.load(f"{wp}")

model = F3Net()
model.load_state_dict(layer_state_dict)
model = model.eval()
model = model.cuda()
test_data = TestReader(dataset_path, mode="train",en_edge=False)
loader = DataLoader(dataset=test_data, batch_size=1, num_workers=0,
                                  shuffle=True, drop_last=True)

img1, img2, label, name = None, None, None, None
color_label = np.array([[0,0,0],[255,255,255]])
for _, (im1, im2, l1, na) in enumerate(loader):
    
    tensor = torch.cat([im1, im2], 1).cuda()
    img1 = im1.squeeze()
    img2 = im2.squeeze()
    label = l1.numpy()
    name = na[0]
    break

print(img1.shape, img2.shape, label.shape)
# rgb_img1 = cv2.imread(os.path.join(dataset_path, f"test/A/{name}"))
# rgb_img1 = normalize(rgb_img1)

# rgb_img1 = cv2.imread(os.path.join(dataset_path, f"test/A/{name}"))
# rgb_img1 = normalize(rgb_img1)



input_tensor = tensor# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
targets = [SemanticSegmentationTarget(1, label)]

target_layers = [model.lkff4]
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
visualization = show_cam_on_image(rgb_img1, grayscale_cam, use_rgb=True)

cv2.imwrite("test.png", visualization)