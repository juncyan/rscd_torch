
import torch
from torchvision import transforms

import os
from PIL import Image
import random
import numpy as np
from torch.utils import data
# from skimage import io
import imageio
import core.datasets.imutils as imutils
import pandas as pd

#label_info = {"0":np.array([0,0,0]), "1":np.array([255,255,255])}
def one_hot_it(label, label_info):
    semantic_map = []
    for info in label_info:
        color = label_info[info].values
        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    return np.stack(semantic_map, axis=-1)


class CDReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(CDReader,self).__init__()
        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)

        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))
        self.label_color = np.array([[1,0],[0,1]])

        self.file_name = []
        self.sst1_images = []
        self.sst2_images = []
        self.gt_images = []
        
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "A", _file))
            self.sst2_images.append(os.path.join(self.path_root, "B", _file))
            self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               
    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        cd_path = self.gt_images[index]

        A_img = np.array(imageio.imread(A_path), np.float32)
        B_img = np.array(imageio.imread(B_path), np.float32)

        gt = np.array(imageio.imread(cd_path))
        
        A_img, B_img, gt = self.__transforms(A_img, B_img, gt)

        if (len(gt.shape) == 3):
            gt = one_hot_it(gt, self.label_info)
        else:
            gt = np.array((gt != 0),dtype=np.int8)
            gt = self.label_color[gt]
        # print(gt)
        gt = np.transpose(np.uint8(gt), [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.float32)

        sst1 = torch.from_numpy(A_img)
        sst2 = torch.from_numpy(B_img)
        if self.aug:
            return sst1, sst2, gt
        return sst1, sst2, gt, self.file_name[index]

    def __len__(self):
        return self.data_num
    
    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list
    
    def __transforms(self, pre_img, post_img, cd_label):
        if self.aug:
            # pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label = imutils.random_fliplr(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_flipud(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_rot(pre_img, post_img, cd_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label
    


class SBCDReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(SBCDReader,self).__init__()
        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)

        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))
        self.label_color = np.array([[1,0],[0,1]])

        self.file_name = []
        self.sst1_images = []
        self.sst2_images = []
        self.gt_images = []
        
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "A", _file))
            self.sst2_images.append(os.path.join(self.path_root, "B", _file))
            self.gt_images.append(os.path.join(self.path_root, "labelA", _file))
            # self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               
    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        cd_path = self.gt_images[index]

        A_img = np.array(imageio.imread(A_path), np.float32)[:,:,:3]
        B_img = np.array(imageio.imread(B_path), np.float32)[:,:,:3]

        gt = np.array(imageio.imread(cd_path))
        
        A_img, B_img, gt = self.__transforms(A_img, B_img, gt)

        gt = np.array((gt > 0), dtype=np.int8)
        gt = self.label_color[gt]

        gt = np.transpose(gt, [2, 0, 1])
        gt = torch.from_numpy(gt).type(torch.float32)

        sst1 = torch.from_numpy(A_img)
        sst2 = torch.from_numpy(B_img)
        if self.aug:
            return sst1, sst2, gt
        return sst1, sst2, gt, self.file_name[index]

    def __len__(self):
        return self.data_num
    
    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list
    
    def __transforms(self, pre_img, post_img, cd_label):
        if self.aug:
            # pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label = imutils.random_fliplr(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_flipud(pre_img, post_img, cd_label)
            pre_img, post_img, cd_label = imutils.random_rot(pre_img, post_img, cd_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label
        
class MultiViewTransform:
    def __init__(self):
        # 定义两种增强强度，一种是全局的(global)，一种是局部的(local)
        # 这在很多方法（如DINO）中被证明是有效的
        self.global_transform_1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)), # 使用支持元组的 torchvision 版本
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.global_transform_2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomSolarize(128, p=0.2), # 随机日晒化
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image):
        # 返回同一张图片的两个不同增强版本
        view1 = self.global_transform_1(image)
        view2 = self.global_transform_2(image)
        return [view1, view2]


if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    x = np.random.random([4,4,3])
    mean = np.std(x, axis=(0,1))
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    print(x)
    print(mean)

    