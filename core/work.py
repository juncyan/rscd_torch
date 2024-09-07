import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import datetime

from .datasets import CDReader, TestReader
from .misc import load_logger
from .misc import train


class Work():
    def __init__(self, model:nn.Module, args, save_dir:str="output"):
        self._seed_init()
        model_name = model.__str__().split("(")[0]
        self.model_name = model_name
        self.args = args
        self.root = save_dir
        self.color_label = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

        os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(self.args.device)
        self.device = torch.device(self.args.device)
        self.model = model.to(self.device, dtype=torch.float)

        self._seed_init()
        self.logger()
        self.dataload()

    def dataload(self, datasetlist=['train', 'val', 'test']):
        train_data = CDReader(self.dataset_path, datasetlist[0], self.args.en_load_edge)
        val_data = CDReader(self.dataset_path, datasetlist[2], self.args.en_load_edge)
        test_data = TestReader(self.dataset_path, datasetlist[2], self.args.en_load_edge)

        self.traindata_num = train_data.__len__()
        self.val_num = val_data.__len__()
        self.test_num =test_data.__len__()

        self.val_loader = DataLoader(dataset=val_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=False, drop_last=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=True, drop_last=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=True, drop_last=True)
        
    def loss(logits, labels):
        if logits.shape == labels.shape:
            labels = torch.argmax(labels,dim=1)
        elif len(labels.shape) == 3:
            labels = labels
        else:
            assert "pred.shape not match label.shape"
        #logits = F.softmax(logits,dim=1)
        return torch.nn.CrossEntropyLoss(dim=1)(logits,labels)
        
    
    def _seed_init(self, seed=32767):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.empty_cache()
        torch.cuda.init()
    
    def logger(self):
        self.dataset_path = '/mnt/data/Datasets/{}'.format(self.args.dataset)
        time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
        self.save_dir = os.path.join('{}/{}'.format(self.root, self.args.dataset.lower()), f"{self.model_name}_{time_flag}")
    
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_model_path = os.path.join(self.save_dir, "{}_best.pth".format(self.model_name))
        log_path = os.path.join(self.save_dir, "train_{}.log".format(self.model_name))
        self.metric_path = os.path.join(self.save_dir, "{}_metrics.csv".format(self.model_name))
        print("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        self.logger = load_logger(log_path)
        self.logger.info("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))

    def __call__(self):
        train(self)

            