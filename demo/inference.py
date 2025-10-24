# Copyright (c) 2023 torchtorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# import cv2
import numpy as np
import torch
import imageio
from torch.nn import functional as F

def bcd_inference(model, images, weight_path=None, save_path="./output/"):
    """
    Launch evalution.

    Args:
        model（nn.Layer): A semantic segmentation model.
        dataset (torch.io.DataLoader): Used to read and process test datasets.
        weights_path (string, optional): weights saved local.
    """


    if weight_path:
        layer_state_dict = torch.load(f"{weight_path}")
        model.load_state_dict(layer_state_dict)
    else:
        exit()
    
    with torch.no_grad():
        pred = model(images)
        pred = torch.argmax(pred, dim=1)
        pred = pred.squeeze().cpu().numpy()
    pred = np.uint8(255 * pred)
    imageio.imwrite(f"{save_path}/bcd_demo.png", np.uint8(pred))


def scd_inference(model, images, weight_path=None, save_path="./output/"):
    
    ST_COLORMAP = np.array([[0,0,0], [0,0,128], [0,128,0], [0,128,128], [128,0,0]])
    ST_CLASSES = ['background', 'water', 'woodland', 'farmland', 'desert']

    if weight_path:
        layer_state_dict = torch.load(f"{weight_path}")
        model.load_state_dict(layer_state_dict)
    else:
        exit()
    
    with torch.no_grad():
        out_change, outputs_A, outputs_B = model(images)
        outputs_A = outputs_A.cpu().detach()
        outputs_B = outputs_B.cpu().detach()

        change_mask = F.sigmoid(out_change).cpu().detach()>0.5 #

        preds_A = torch.argmax(outputs_A, dim=1)
        preds_B = torch.argmax(outputs_B, dim=1)
        preds_A = (preds_A*change_mask.squeeze().long())
        preds_B = (preds_B*change_mask.squeeze().long())
        preds_A = preds_A.squeeze().cpu().numpy()
        preds_B = preds_B.squeeze().cpu().numpy()
        change_mask = change_mask.squeeze().cpu().numpy()

    pred = np.uint8(255 * change_mask)
    is1 = ST_COLORMAP[preds_A]
    is2 = ST_COLORMAP[preds_B]
    imageio.imwrite(f"{save_path}/bcd_demo.png", np.uint8(pred))
    imageio.imwrite(f"{save_path}/sa_demo.png", np.uint8(is1))
    imageio.imwrite(f"{save_path}/sb_demo.png", np.uint8(is2))
      
