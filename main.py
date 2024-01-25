# 调用官方库及第三方库
import torch
from torch.utils.data import DataLoader
import numpy as np
import datetime
import platform
import random
import os

# 基础功能
from dataset.CDReader import CDReader, TestReader
from work.train import train

# 模型导入
from core.models.unet import UNet
from core.models.deeplabv3_plus import DeepLabV3Plus
from core.models.pspnet import PSPNet
from core.models.denseaspp import DenseASPP
from core.models.hrnet import HRNet
from core.models.dfanet import DFANet
from core.models.fcn import FCN32s
from cd_models.mscanet.model import MSCANet
from cd_models.aernet import AERNet
from cd_models.ResUnet import ResUnet
from cd_models.a2net import LightweightRSCDNet
from cd_models.ussfcnet.ussfcnet import USSFCNet
from cd_models.dtcdscn import DTCDSCNet
from cd_models.changeformer import ChangeFormerV6
from cd_models.dminet import DMINet
from cd_models.siamunet_diff import SiamUnet_diff
from cd_models.siamunet import SiamUnet_conc
from cd_models.SUNet import SUNnet
from cd_models.dsamnet import DSAMNet
from cd_models.stanet import STANetSA
from cd_models.icifnet import ICIFNet
from cd_models.dsifn import DSIFN
from cd_models.bit_cd import BIT_CD
from cd_models.transunet import TransUNet
from cd_models.bisrnet import BiSRNet

from common import Args


# class parameter:
#     lr = params["lr"]
#     momentum = params["momentum"]
#     weight_decay = params["weight_decay"]
#     num_epochs = num_epochs
#     batch_size = batch_size

# dataset_name = "GVLM_CD_d"
# dataset_name = "LEVIR_d"
# dataset_name = "LEVIR_c"
dataset_name = "CLCD"
# dataset_name = "SYSCD_d"
dataset_path = '/mnt/data/Datasets/{}'.format(dataset_name)
num_classes = 2
batch_size = 4
num_epochs = 100 


def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    # 代码运行预处理
    seed_torch(32765)
    torch.cuda.empty_cache()
    torch.cuda.init()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # mode:["train","eval","test"] or [1,2,3]

    # 模型选择
    # model = DeepLabV3Plus(6, 2, backbone="xception", pretrained_base=False,dilated=True)
    # model = DenseASPP(num_classes, pretrained_base=False)
    # model = LEDNet(args.num_classes,"resnet50")
    # model = BiSeNet(args.num_classes)
    # model = DFANet(num_classes, norm_layer=torch.nn.BatchNorm2d)
    # model = UNet(6, num_classes)
    # model = PSPNet(num_classes, pretrained_base=False)
    # model = HRNet(num_classes)
    # model = FCN32s(num_classes, aux=True, pretrained_base=False)
    # model = CDNet(img_size=512)
    # model = LightweightRSCDNet()
    # model = USSFCNet(in_ch=3)
    # model = DTCDSCNet()
    # model = ChangeFormerV6()
    # model = DMINet()
    # model = ResUnet()
    # model = SUNnet(4,out_size=[256,256])
    # model = DSAMNet(n_class=2)
    # model = STANetSA(3)
    # model = ICIFNet(2)
    # model = DSIFN()
    model = BiSRNet()

    model_name = model.__str__().split("(")[0]
    args = Args('output/{}'.format(dataset_name.lower()), model_name)
    args.data_name = dataset_name
    args.num_classes = num_classes
    args.batch_size = batch_size
    args.iters = num_epochs
    args.pred_idx = 0

    # pred_data = Reader_Only_Image()
    eval_data = CDReader(path_root = dataset_path, mode="val", en_edge=False)
    train_data = CDReader(path_root = dataset_path, mode="train", en_edge=False)
    
    # dataloader_pred = DataLoader(pred_data, batch_size, num_workers=1)
    dataloader_eval = DataLoader(dataset=eval_data, batch_size=args.batch_size, num_workers=16,
                                 shuffle=False, drop_last=True)
    dataloader_train = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=16,
                                  shuffle=True, drop_last=True)
    
    test_data = TestReader(path_root = dataset_path, mode="test", en_edge=False)
    dataloader_test = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=0,
                                  shuffle=True, drop_last=True)
    
    # try:
    #     model.load_state_dict(torch.load(save_model_dir))
    #     print("load success")
    # except:
    #     args.num_epochs = 300
    #     args.params["lr"] = 0.0005
    model = model.to('cuda', dtype=torch.float)
    # model.load_state_dict(torch.load("/home/jq/Code/torch/output/levir_d/SiamUnet_diff_2023_10_26_16/SiamUnet_diff_best.pth"))
    train(model, dataloader_train, dataloader_eval, dataloader_test, args)
    