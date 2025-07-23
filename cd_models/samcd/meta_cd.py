import torch
from torch import nn
from .fastsam import FastSAM
from torch.nn import functional as F
from torch.autograd import Function
from torch.autograd import Variable
from typing import Dict, List
from torchvision import models

from .utils.misc import initialize_weights
from .utils.loss import LatentSimilarity


criterion_sem = LatentSimilarity(T=3.0)

class BinarizedF(Function):
  @staticmethod
  def forward(ctx, input, threshold):
    device = input.device
    ctx.save_for_backward(input,threshold)
    a = torch.ones_like(input).to(device)
    b = torch.zeros_like(input).to(device)
    output = torch.where(input>=threshold,a,b)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    # print('grad_output',grad_output)
    input,threshold = ctx.saved_tensors
    grad_input = grad_weight  = None

    if ctx.needs_input_grad[0]:
      grad_input= 0.2*grad_output
    if ctx.needs_input_grad[1]:
      grad_weight = -grad_output
    return grad_input, grad_weight


class compressedSigmoid(nn.Module):
    def __init__(self, para=2.0, bias=0.2):
        super(compressedSigmoid, self).__init__()

        self.para = para
        self.bias = bias

    def forward(self, x):
        output = 1. / (self.para + torch.exp(-x)) + self.bias
        return output

class BinarizedModule(nn.Module):
  def __init__(self, input_channels=720):
    super(BinarizedModule, self).__init__()

    self.Threshold_Module = nn.Sequential(
        nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.PReLU(),
        # nn.AvgPool2d(15, stride=1, padding=7),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.PReLU(),
        # nn.AvgPool2d(15, stride=1, padding=7),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.PReLU(),
        nn.AvgPool2d(15, stride=1, padding=7),
        nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
        nn.AvgPool2d(15, stride=1, padding=7),
    )
    '''这里面放的是chatgpt改进版阈值学习
      self.Threshold_Module = nn.Sequential(nn.Conv2d(input_channels, 256, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.PReLU(),
        nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.PReLU(),
        nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.PReLU(),
        nn.AvgPool2d(15, stride=1, padding=7),
        nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),)'''
    
    self.sig = compressedSigmoid()

    self.weight = nn.Parameter(torch.Tensor(1).fill_(0.5),requires_grad=True)
    self.bias = nn.Parameter(torch.Tensor(1).fill_(0), requires_grad=True)
    self.sigmoid = nn.Sigmoid()

  def forward(self,feature, pred_map):


    p = F.interpolate(pred_map.detach(), scale_factor=0.125)
    f = F.interpolate(feature.detach(), scale_factor=0.5)
    # import pdb
    # pdb.set_trace()
    f = f * p
    threshold = self.Threshold_Module(f)

    threshold = self.sig(threshold *10.) # fixed factor

    threshold = F.interpolate(threshold, scale_factor=8)
    #这里添加了sigmoid的修改
    Binar_map = BinarizedF.apply(pred_map, threshold)
    return threshold, Binar_map



class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.cnn = nn.Conv2d(1,1,3,stride=1,padding=1)

  def forward(self,input):
    output =self.cnn(input)
    return output
  
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class Space_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Space_Attention, self).__init__()
        self.SA = nn.Sequential( 
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction, momentum=0.95),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()        
        A = self.SA(x)
        return A

class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SAM_CD_forward(nn.Module):
    def __init__(
        self,
        num_embed=8,
        model_name: str='/home/jq/Code/SAM-CD/FastSAM-x.pt', #model_pt_path
        
        device=0,
        conf: float=0.4,
        iou: float=0.9,
        imgsz: int=256,
        retina_masks: bool=True,
        ):
        super(SAM_CD_forward, self).__init__()
        self.model = FastSAM(model_name)
        self.device = device
        self.retina_masks = retina_masks
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.image = None
        self.image_feats = None        
         
        self.Adapter32 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter16 = nn.Sequential(nn.Conv2d(640, 160, kernel_size=1, stride=1, padding=0, bias=False),
                                       nn.BatchNorm2d(160), nn.ReLU())
        self.Adapter8 = nn.Sequential(nn.Conv2d(320, 80, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(80), nn.ReLU())
        self.Adapter4 = nn.Sequential(nn.Conv2d(160, 40, kernel_size=1, stride=1, padding=0, bias=False),
                                      nn.BatchNorm2d(40), nn.ReLU())
                                       
        self.Dec2 = _DecoderBlock(160, 160, 80)
        self.Dec1 = _DecoderBlock(80, 80, 40)  
        self.Dec0 = _DecoderBlock(40, 40, 64)
        
        self.SA = Space_Attention(16, 16, 4)
        self.segmenter = nn.Conv2d(64, num_embed, kernel_size=1)        
        self.resCD = self._make_layer(ResBlock, 128, 128, 6, stride=1)
        self.headC = nn.Sequential(nn.Conv2d(128, 16, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.segmenterC = nn.Conv2d(16, 1, kernel_size=1)
        self.resnet18 = models.resnet34(pretrained=True)
        self.alpha = nn.Parameter(torch.zeros(1)).cuda(device)
        self.Sigmoid = nn.Sigmoid()
        self.layer1 = nn.Sequential(*list(self.resnet18.children())[:5])  # 64 channels
        self.layer2 = self.resnet18.layer2  # 128 channels
        self.layer3 = self.resnet18.layer3  # 256 channels
        self.layer4 = self.resnet18.layer4  # 512 channels
        # Additional layers to adjust feature sizes
        self.conv1 = nn.Conv2d(128, 320, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 640, kernel_size=1)
        self.conv3 = nn.Conv2d(512, 640, kernel_size=1)
        self.conv4 = nn.Conv2d(64, 160, kernel_size=1)                                
        for param in self.model.model.parameters():
            param.requires_grad = False
        initialize_weights(self.Adapter32, self.Adapter16, self.Adapter8, self.Adapter4, self.Dec2, self.Dec1, self.Dec0,\
                           self.segmenter, self.resCD, self.headC, self.segmenterC)

    def run_encoder(self, image):
        self.image = image
        cnn_feat1 = self.layer1(self.image) #torch.Size([4, 64, 128, 128])
        cnn_feat2 = self.layer2(cnn_feat1) # torch.Size([4, 128, 64, 64])
        cnn_feat3 = self.layer3(cnn_feat2) # torch.Size([4, 256, 32, 32])
        cnn_feat4 = self.layer4(cnn_feat3) # torch.Size([4, 512, 16, 16])
        # Adjust feature sizes
        cnn_feat1 = self.conv4(cnn_feat1)#torch.Size([4, 64, 128, 128])
        cnn_feat2 = self.conv1(cnn_feat2)#torch.Size([4, 320, 64,64])
        cnn_feat3 = self.conv2(cnn_feat3)#torch.Size([4, 640, 32, 32])
        cnn_feat4 = self.conv3(cnn_feat4)#torch.Size([4, 640,16, 16])
        cnn_feats = [cnn_feat2,cnn_feat3,cnn_feat4,cnn_feat1]

        feats = self.model(
            self.image,
            device=f"cuda:{self.device}",
            retina_masks=self.retina_masks,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou
            )
        self.gate = self.Sigmoid(self.alpha)

        feats = [self.gate*x + (1 - self.gate)*(y.clone().cuda(self.device)) for x, y in zip(cnn_feats,feats)]
        return feats

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
    
        input_shape = x1.shape[-2:]
        featsA = self.run_encoder(x1)
        featsB = self.run_encoder(x2)
        
        featA_s4 = self.Adapter4(featsA[3])
        featA_s8 = self.Adapter8(featsA[0])
        featA_s16 = self.Adapter16(featsA[1])
        featA_s32 = self.Adapter32(featsA[2])
        
        decA_2 = self.Dec2(featA_s32, featA_s16)
        decA_1 = self.Dec1(decA_2, featA_s8)
        decA_0 = self.Dec0(decA_1, featA_s4)
        outA = self.segmenter(decA_0)
        
        featB_s4 = self.Adapter4(featsB[3])
        featB_s8 = self.Adapter8(featsB[0])
        featB_s16 = self.Adapter16(featsB[1])
        featB_s32 = self.Adapter32(featsB[2])              
        
        decB_2 = self.Dec2(featB_s32, featB_s16)
        decB_1 = self.Dec1(decB_2, featB_s8)
        decB_0 = self.Dec0(decB_1, featB_s4)
        outB = self.segmenter(decB_0)
             
        A = self.SA(torch.cat([outA, outB], dim=1))  
        featC = torch.cat([decA_0, decB_0], 1)
        featC = self.resCD(featC)
        featC = self.headC(featC) * A
        outC = self.segmenterC(featC)
        outC = F.interpolate(outC, input_shape, mode="bilinear", align_corners=True)
        return outC,\
               F.interpolate(outA, input_shape, mode="bilinear", align_corners=True),\
               F.interpolate(outB, input_shape, mode="bilinear", align_corners=True),\
               featC
    
class Meta_CD(nn.Module):
#     @ARTICLE{10902491,
#   author={Gao, Junyu and Zhang, Da and Wang, Feiyu and Ning, Lichen and Zhao, Zhiyuan and Li, Xuelong},
#   journal={IEEE Transactions on Geoscience and Remote Sensing}, 
#   title={Combining SAM With Limited Data for Change Detection in Remote Sensing}, 
#   year={2025},
#   volume={63},
#   number={},
#   pages={1-11},
#   keywords={Feature extraction;Adaptation models;Remote sensing;Data models;Convolutional neural networks;Training;Image segmentation;Transformers;Decoding;Computational modeling;Change detection;foundation model;limited data;remote sensing},
#   doi={10.1109/TGRS.2025.3545040}}

    def __init__(self, dev=0):
        super(Meta_CD, self).__init__()
        device = f"cuda:{dev}"
        self.sam_1 = SAM_CD_forward(device=dev).cuda(dev)
        self.Binar = BinarizedModule(input_channels=16)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        outputs, outA, outB ,featC= self.sam_1(x1, x2)
        pre_map = self.Sigmoid(outputs)
        threshold_matrix, binar_map = self.Binar(featC,pre_map)
        return outputs, outA, outB, binar_map, threshold_matrix
    
    @staticmethod
    def loss(preds, label):
        outputs, outA, outB ,binar_map, threshold_map = preds
        device = outputs.device
        criterion_sem.to(device)
        lab = torch.argmax(label,1,True).float()
        loss_bn = F.binary_cross_entropy_with_logits(outputs, lab)
        loss_t = criterion_sem(outA, outB, lab)
            #pdb.set_trace()
        loss_b = (torch.abs(binar_map-lab)).mean()
        loss = loss_bn + loss_t + loss_b
        return loss

    @staticmethod
    def predict(preds):
        pred = preds[0]
        p = F.sigmoid(pred)
        p = (p > 0.5)
        p = p.int()
        return p
