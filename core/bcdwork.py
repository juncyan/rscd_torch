import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import datetime

from .datasets import CDReader
from .cdmisc import load_logger
from .cdmisc import train


class Work():
    def __init__(self, model:nn.Module, args):
        self._seed_init()
        model_name = model.__str__().split("(")[0]
        self.args = args
        self.args.model = model_name

        self.color_label = np.array([[0,0,0],[255,255,255],[0,128,0],[0,0,128]])

        os.environ['CUDA_VISIBLE_DEVICES'] = "{}".format(self.args.device)
        self.model = model.to(self.args.device, dtype=torch.float)

        self._seed_init()
        self.logger()
        self.dataloader()

        train(model, self.train_loader, self.val_loader, self.test_loader, self.args)

    def dataloader(self, datasetlist=['train', 'val', 'test']):
        train_data = CDReader(self.dataset_path, datasetlist[0])
        val_data = CDReader(self.dataset_path, datasetlist[2])
        test_data = CDReader(self.dataset_path, datasetlist[2])

        self.args.traindata_num = train_data.__len__()
        self.args.val_num = val_data.__len__()
        self.args.test_num =test_data.__len__()

        self.val_loader = DataLoader(dataset=val_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=False, drop_last=True)
        self.train_loader = DataLoader(dataset=train_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=True, drop_last=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=self.args.batch_size, num_workers=self.args.num_workers,
                                    shuffle=True, drop_last=True)
        
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
        self.save_dir = os.path.join('{}/{}'.format(self.args.root, self.args.dataset.lower()), f"{self.args.model}_{time_flag}")
    
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.args.best_model_path = os.path.join(self.save_dir, "{}_best.pth".format(self.args.model))
        log_path = os.path.join(self.save_dir, "train_{}.log".format(self.args.model))
        self.args.metric_path = os.path.join(self.save_dir, "{}_metrics.csv".format(self.args.model))
        self.args.save_dir = self.save_dir
        print("log save at {}, metric save at {}, weight save at {}".format(log_path, self.args.metric_path, self.args.best_model_path))
        self.args.logger = load_logger(log_path)
        self.log_misc()
        self.args.logger.info("log save at {}, metric save at {}, weight save at {}".format(log_path, self.args.metric_path, self.args.best_model_path))
    
    def log_misc(self):
        if self.args.logger == None:
            return
        self.args.logger.info("Model {}, Datasets {}".format(self.args.model, self.args.dataset))
        self.args.logger.info("lr {}, batch_size {}".format(str(self.args.lr), self.args.batch_size))


    def __call__(self):
        train(self.model, self.train_loader, self.val_loader, self.test_loader, self.args)

            