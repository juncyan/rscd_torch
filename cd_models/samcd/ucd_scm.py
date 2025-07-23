import torch
from torch import nn

import cv2
import numpy as np

from .utils.cal_cos import otsu_thres, cal_cos_smilarity_float
from .utils.clip import FastSAMPrompt
from .fastsam.model import FastSAM
from .utils.misc import initialize_weights
from .utils.loss import LatentSimilarity
import clip

class UCD_SCM(nn.Module):
    def __init__(
        self,
        device=0,
        imgsz: int=256,
        retina_masks: bool=True,
        ):
        super().__init__()
        self.model_sam = FastSAM('/home/jq/Code/weights/FastSAM-x.pt')
        self.device = f"cuda:{device}"
        self.imgsz = imgsz
        self.image = None
        self.image_feats = None  
        self.model_clip, self.clip_preprocess = clip.load('/home/jq/Code/weights/ViT-B-32.pt',
                                                 device=self.device)      

    def run_encoder(self, image):   
        feats = self.model_sam(
            image,
            device=self.device,
            retina_masks=True,
            imgsz=self.imgsz,
            conf=0.4,
            iou=0.9
            )
        return feats

    def forward(self, x1, x2=None):
        if x2 is None:
            x2 = x1[:, 3:, :, :]
            x1 = x1[:, :3, :, :]
        featsA = self.run_encoder(x1)[0]
        featsB = self.run_encoder(x2)[0]
        # print(featsA[0].shape, featsB[0].shape)
        dm_cons_cossim = recalibrated_feature_fusion(featsA, featsB)
        dm_cons_cosdis = 1 - dm_cons_cossim
        dm_cons_cosdis = np.clip(dm_cons_cosdis, 0, 1).astype(np.float32).squeeze()
        prev_prompt_process = FastSAMPrompt(x1, None, device=self.device)
        prev_bld_score = prev_prompt_process.text_prompt(clip_model=self.model_clip, 
                                                         preprocess=self.clip_preprocess)
        curr_prompt_process = FastSAMPrompt(x2, None, device=self.device)
        curr_bld_score = curr_prompt_process.text_prompt(clip_model=self.model_clip, 
                                                         preprocess=self.clip_preprocess)

        mean_bld_score = (prev_bld_score + curr_bld_score) / 2. # float.
        conc_bld_score = np.concatenate([np.expand_dims(prev_bld_score, 2), np.expand_dims(curr_bld_score, 2)], 2)
        mean_bld_score = np.max(conc_bld_score, axis=2) # 01

        bld_mask = np.where(mean_bld_score>=0.5, 1, 0).astype(np.float32) # 1
        nonbld_mask = np.where(mean_bld_score<0.5, mean_bld_score, 0).astype(np.float32) # 0-0.5
        strec_nonbld_mask = nonbld_mask * 2 # 0-1
        whole_bld_mask = bld_mask + strec_nonbld_mask

        # mul
        dm_cons_cosdis = np.multiply(dm_cons_cosdis, whole_bld_mask)
        return dm_cons_cosdis

def rf_module(in_feats):
    sque_feats = np.ones(in_feats.shape[2])
    for c in range(in_feats.shape[2]):
        sque_feats[c] = np.mean(in_feats[:,:,c])
    
    for i, ave in np.ndenumerate(sque_feats):
        in_feats[:,:,i[0]] = in_feats[:,:,i[0]] * ave
    exit_feats = in_feats
    return exit_feats     

def recalibrated_feature_fusion(prev_hier_feats, curr_hier_feats):
    
    '''
    Feature dimensions extracted from fastsam:
    '''
    # ori
        # level1 (1, 320, 128, 128)
        # level2 (1, 640, 64, 64)
        # level3 (1, 640, 32, 32)
        # level0 (1, 160, 256, 256)
    # reform_prev_hier_feats
        # level0 (1, 160, 256, 256)
        # level1 (1, 320, 128, 128)
        # level2 (1, 640, 64, 64)
        # level3 (1, 640, 32, 32)

    reform_prev_hier_feats, reform_curr_hier_feats = [], []
    # whole format
    for ind in [3,0,1,2]:
    # for ind in [0,1,2]:
        prev_hier_feat_tensor, curr_hier_feat_tensor = prev_hier_feats[ind], curr_hier_feats[ind]
        prev_hier_feat_arr, curr_hier_feat_arr = prev_hier_feat_tensor.cpu().numpy(), curr_hier_feat_tensor.cpu().numpy()
        prev_hier_feat_arr = prev_hier_feat_arr[0].transpose([1,2,0]) # out: channel last
        curr_hier_feat_arr = curr_hier_feat_arr[0].transpose([1,2,0])
        reform_prev_hier_feats.append(prev_hier_feat_arr)
        reform_curr_hier_feats.append(curr_hier_feat_arr)
    

    # stage3
    c3_prev_feats, c3_curr_feats = reform_prev_hier_feats[3], reform_curr_hier_feats[3] # 16,16,640
    c3_prev_feats, c3_curr_feats = rf_module(c3_prev_feats), rf_module(c3_curr_feats) # se
    c3_prev_feats_160 = c3_prev_feats[:,:,list(range(0,c3_prev_feats.shape[2],c3_prev_feats.shape[2]//160))] # 16,16,160
    c3_curr_feats_160 = c3_curr_feats[:,:,list(range(0,c3_curr_feats.shape[2],c3_curr_feats.shape[2]//160))]
    p3_prev_feats = cv2.resize(c3_prev_feats_160, (c3_prev_feats_160.shape[0]*2,c3_prev_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR) # 32,32,160
    p3_curr_feats = cv2.resize(c3_curr_feats_160, (c3_curr_feats_160.shape[0]*2,c3_curr_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR)

    # stage2
    c2_prev_feats, c2_curr_feats = reform_prev_hier_feats[2], reform_curr_hier_feats[2] # 32,32,640
    c2_prev_feats, c2_curr_feats = rf_module(c2_prev_feats), rf_module(c2_curr_feats) # se
    c2_prev_feats_160 = c2_prev_feats[:,:,list(range(0,c2_prev_feats.shape[2],c2_prev_feats.shape[2]//160))] # 32,32,160
    c2_curr_feats_160 = c2_curr_feats[:,:,list(range(0,c2_curr_feats.shape[2],c2_curr_feats.shape[2]//160))]
    # c2 + p3
    c2_prev_feats_160 = c2_prev_feats_160 + p3_prev_feats # 32,32,160
    c2_curr_feats_160 = c2_curr_feats_160 + p3_curr_feats
    # p2
    p2_prev_feats = cv2.resize(c2_prev_feats_160, (c2_prev_feats_160.shape[0]*2,c2_prev_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR) # 64,64,160
    p2_curr_feats = cv2.resize(c2_curr_feats_160, (c2_curr_feats_160.shape[0]*2,c2_curr_feats_160.shape[1]*2), interpolation=cv2.INTER_LINEAR)

    # stage1
    c1_prev_feats, c1_curr_feats = reform_prev_hier_feats[1], reform_curr_hier_feats[1] # 64,64,320
    c1_prev_feats, c1_curr_feats = rf_module(c1_prev_feats), rf_module(c1_curr_feats) # se
    c1_prev_feats_160 = c1_prev_feats[:,:,list(range(0,c1_prev_feats.shape[2],c1_prev_feats.shape[2]//160))] # 64,64,160
    c1_curr_feats_160 = c1_curr_feats[:,:,list(range(0,c1_curr_feats.shape[2],c1_curr_feats.shape[2]//160))]
    # c1 + p2
    c1_prev_feats_160 = c1_prev_feats_160 + p2_prev_feats # 64,64,160
    c1_curr_feats_160 = c1_curr_feats_160 + p2_curr_feats
    


    # s
    s3_prev_128_160 = cv2.resize(c3_prev_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s3_curr_128_160 = cv2.resize(c3_curr_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s2_prev_128_160 = cv2.resize(c2_prev_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s2_curr_128_160 = cv2.resize(c2_curr_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s1_prev_128_160 = cv2.resize(c1_prev_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    s1_curr_128_160 = cv2.resize(c1_curr_feats_160, (256,256), interpolation=cv2.INTER_LINEAR)
    # concate s
    cons_prev_feats = np.concatenate([s3_prev_128_160, s2_prev_128_160, s1_prev_128_160], axis=2)
    cons_curr_feats = np.concatenate([s3_curr_128_160, s2_curr_128_160, s1_curr_128_160], axis=2)

    # upsam->dm concate
    cons_prev_feats_512 = np.concatenate([cv2.resize(cons_prev_feats[:,:,:320], (1024,1024), interpolation=cv2.INTER_LINEAR), cv2.resize(cons_prev_feats[:,:,320:], (1024,1024), interpolation=cv2.INTER_LINEAR)], axis=2) 
    cons_curr_feats_512 = np.concatenate([cv2.resize(cons_curr_feats[:,:,:320], (1024,1024), interpolation=cv2.INTER_LINEAR), cv2.resize(cons_curr_feats[:,:,320:], (1024,1024), interpolation=cv2.INTER_LINEAR)], axis=2)
    

    dm_cons_cossim = cal_cos_smilarity_float(cons_prev_feats_512, cons_curr_feats_512)
    return dm_cons_cossim