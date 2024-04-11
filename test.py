from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from f3net.f3net import F3Net


# wp = r"/mnt/data/TrainLog_LEVIR_GVLM_CLCD/gvlm_cd/LKAUChange_2023_11_09_10/LKAUChange_best.pdparams"
model = F3Net()

target_layers = [model.layer4[-1]]
input_tensor = 1# Create an input tensor image for your model..
# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers)

targets = [ClassifierOutputTarget(281)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

rgb_img = 0
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

model_outputs = cam.outputs