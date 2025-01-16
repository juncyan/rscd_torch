import torch
import torch.nn.functional as F

from scdmodel.scd import SCDSam_Mamba
from models.cdfa import ConDSeg
from cd_models.mobilesam import build_sam_vit_t
from scdmodel.cddecoder import CD_Mamba
from scdmodel.decoder import SemantiCrossA
from models.cdfa import ContrastDrivenFeatureAggregation

from einops import rearrange, repeat




if __name__ == "__main__":
    print('test')
    x1 = torch.randn([1,160,64,64]).cuda()
    x2 = torch.randn([1,320,8,8]).cuda()
    a = torch.randn([1,256,16,16]).cuda()
    m = SemantiCrossA(320, 160, 64, 512).cuda()
    y = m(x1,x2,a)
    for i in y:
        print(i.shape)