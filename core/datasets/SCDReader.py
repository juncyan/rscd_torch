import torch
import torchvision.transforms as tfs
import os
# from skimage import io
import imageio
import random
import numpy as np
from torch.utils import data
import pandas as pd
import core.datasets.imutils as imutils

num_classes = 7
ST_COLORMAP = [[255,255,255], [0,0,255], [128,128,128], [0,128,0], [0,255,0], [128,0,0], [255,0,0]]
ST_CLASSES = ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field']

class SCDReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(SCDReader,self).__init__()
        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)

        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)

        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))

        self.file_name = []
        self.sst1_images = []
        self.sst1_lab = []
        self.sst2_images = []
        self.sst2_lab = []
        # self.gt_images = []
        
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "A", _file))
            self.sst2_images.append(os.path.join(self.path_root, "B", _file))
            self.sst1_lab.append(os.path.join(self.path_root, "labelA", _file))
            self.sst2_lab.append(os.path.join(self.path_root, "labelB", _file))
            # self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               
    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        labA_path = self.sst1_lab[index]
        labB_path = self.sst2_lab[index]
        # cd_path = self.gt_images[index]

        A_img = np.array(imageio.imread(A_path, pilmode="RGB"), np.float32)
        B_img = np.array(imageio.imread(B_path, pilmode="RGB"), np.float32)
        labA = np.array(imageio.imread(labA_path), np.float32)
        labB = np.array(imageio.imread(labB_path), np.float32)

        # if os.path.exists(cd_path):
        #     cd = np.array(io.imread(cd_path), np.float32)
        #     cd = cd / 255.
        # else:
        #     cd = np.array(labA > 0, np.float32)
        
        A_img, B_img, labA, labB = self.__transforms(A_img, B_img, labA, labB)

        sst1 = torch.from_numpy(A_img)
        sst2 = torch.from_numpy(B_img)
        cd = torch.from_numpy(np.array(labA > 0, np.float32))
        gt1 = torch.from_numpy(labA)
        gt2 = torch.from_numpy(labB)

        if self.aug:
            return sst1, sst2, gt1, gt2, cd
        return sst1, sst2, gt1, gt2, cd, self.file_name[index]

    def __len__(self):
        return self.data_num
    
    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list
    
    def __transforms(self, pre_img, post_img, t1_label, t2_label):
        if self.aug:
            # pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, t1_label, t2_label)
            pre_img, post_img, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, t1_label, t2_label)
            pre_img, post_img, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, t1_label, t2_label

class MusReader(data.Dataset):
    def __init__(self,path_root="./dataset/",mode="train"):
        super(MusReader,self).__init__()

        self.aug = mode == "train"
        self.path_root = os.path.join(path_root, mode)
        self.data_list = self._get_list(self.path_root)
        self.data_num = len(self.data_list)
        self.label_info = pd.read_csv(os.path.join(path_root, 'label_info.csv'))

        self.file_name = []
        self.sst1_images = []
        self.sst1_lab = []
        self.sst2_images = []
        self.sst2_lab = []
        self.gt_images = []
        
        for _file in self.data_list:
            self.sst1_images.append(os.path.join(self.path_root, "A", _file))
            self.sst2_images.append(os.path.join(self.path_root, "B", _file))
            self.sst1_lab.append(os.path.join(self.path_root, "labelA", _file))
            self.sst2_lab.append(os.path.join(self.path_root, "labelB", _file))
            self.gt_images.append(os.path.join(self.path_root, "label", _file))
            self.file_name.append(_file)
               

    def __getitem__(self, index):

        A_path = self.sst1_images[index]
        B_path = self.sst2_images[index]
        labA_path = self.sst1_lab[index]
        labB_path = self.sst2_lab[index]
        # cd_path = self.gt_images[index]

        A_img = np.array(imageio.imread(A_path, pilmode="RGB"), np.float32)
        B_img = np.array(imageio.imread(B_path, pilmode="RGB"), np.float32)
  

        labA = np.array(imageio.imread(labA_path), np.uint8)
        labA = imutils.one_hot_it(labA, self.label_info)
        labA = np.array(np.argmax(labA, -1), np.float32)

        labB = np.array(imageio.imread(labB_path), np.uint8)
        labB = imutils.one_hot_it(labB, self.label_info)
        labB = np.array(np.argmax(labB, -1), np.float32)
        
        # gtcd = np.array(io.imread(cd_path), np.uint8)
        # gtcd = gtcd / 255.
        gtcd = np.array(labA > 0, np.float32)
        
        sst1, sst2, gt1, gt2, gtcd = self.__transforms(self.aug, A_img, B_img, labA, labB, gtcd)
    
        sst1 = torch.from_numpy(sst1)
        sst2 = torch.from_numpy(sst2)
        gt1 = torch.from_numpy(gt1)
        gt2 = torch.from_numpy(gt2)
        gtcd = torch.from_numpy(gtcd)
        # print(sst1.shape, sst2.shape, gt1.shape, gt2.shape, gtcd.shape)
        if self.aug:
            return sst1, sst2, gt1, gt2, gtcd
        return sst1, sst2, gt1, gt2, gtcd, self.file_name[index]

    def __len__(self):
        return self.data_num

    def _get_list(self, list_path):
        data_list = os.listdir(os.path.join(list_path,'A'))
        return data_list
    
    def __transforms(self, aug, pre_img, post_img, cd_label, t1_label, t2_label):
        if aug:
            # pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_crop_mcd(pre_img, post_img, cd_label, t1_label, t2_label, self.crop_size)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_fliplr_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_flipud_mcd(pre_img, post_img, cd_label, t1_label, t2_label)
            pre_img, post_img, cd_label, t1_label, t2_label = imutils.random_rot_mcd(pre_img, post_img, cd_label, t1_label, t2_label)

        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))
        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, cd_label, t1_label, t2_label   



if __name__ == "__main__":
    dataset_path = '/mnt/data/Datasets/CLCD'
    x = np.random.random([4,4,3])
    mean = np.std(x, axis=(0,1))
    mean = np.array(mean)[np.newaxis, np.newaxis, :]
    print(x)
    print(mean)

    