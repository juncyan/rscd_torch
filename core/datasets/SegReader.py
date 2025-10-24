import torch
from torchvision import transforms
import os
# from skimage import io
import imageio
import random
import numpy as np
from torch.utils import data
import pandas as pd
import core.datasets.imutils as imutils
from PIL import Image

num_classes = 7

class SegReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(SegReader,self).__init__()
        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)

        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))

        self.file_name = []
        self.sst_images = []
        self.gt_images = []
        
        
        for _file in self.data_list:
            self.sst_images.append(os.path.join(self.path_root, "image", _file))
            self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               
    def __getitem__(self, index):

        A_path = self.sst_images[index]
        cd_path = self.gt_images[index]

        sst = np.array(imageio.imread(A_path), np.float32)
        gt = np.array(imageio.imread(cd_path), np.float32)
        
        # print(self.label_info)
        if len(gt.shape) == 3:
            gt = imutils.one_hot_it(gt, self.label_info)
            gt = np.argmax(gt, axis=-1)
        gt = np.int64(gt)
        sst, gt = self.__transforms(sst, gt)
        
        sst = torch.from_numpy(sst) 
        gt = torch.from_numpy(gt)

        if self.aug:
            return sst, gt
        return sst, gt, self.file_name[index]

    def __len__(self):
        return self.data_num
    
    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'image'))
        return data_list
    
    def __transforms(self, sst, gt):
        if self.aug:
            sst, gt = imutils.random_fliplr_seg(sst, gt)
            sst, gt = imutils.random_flipud_seg(sst, gt)
            sst, gt = imutils.random_rot_seg(sst, gt)

        sst = imutils.normalize_img(sst)  # imagenet normalization
        sst = np.transpose(sst, (2, 0, 1))
        # gt = np.transpose(gt, (2, 0, 1))

        return sst, gt


class UnLabelSegReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(UnLabelSegReader,self).__init__()
        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)

        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)

        # self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))

        base = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]
        self.transform = transforms.Compose(base)

        
        self.file_name = []
        self.sst_images = []
        
        for _file in self.data_list:
            self.sst_images.append(os.path.join(self.path_root, "image", _file))
               
    def __getitem__(self, index):
        A_path = self.sst_images[index]
        sst = Image.open(A_path)#np.array(imageio.imread(A_path), np.float32)
        ss1 = self.transform(sst)
        ss2 = self.transform(sst)
        return ss1, ss2
    
    def __len__(self):
        return self.data_num
    
    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'image'))
        return data_list

   


if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    x = np.random.random([4,4,3])
    mean = np.std(x, axis=(0,1))
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    print(x)
    print(mean)

    