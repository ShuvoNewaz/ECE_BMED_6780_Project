from src.models.esfpnet.esfpnet import ESFPNetStructure
from src.models.pspnet.pspnet import psp_model_optimizer
from src.models.unet.unet_model import UNet#Dummy as UNet
import torch

def get_model_optimizer(model_name, B):
    if model_name == 'esfpnet':
        lung_segmenter = ESFPNetStructure(B, 160, 0.2)
        optimizer1 = torch.optim.AdamW(lung_segmenter.parameters(), lr=1e-4)
        infection_segmenter = ESFPNetStructure(B, 160, 0.2)
        optimizer2 = torch.optim.AdamW(infection_segmenter.parameters(), lr=1e-4)
    elif model_name == 'unet':
        # pretrained_unet = torch.load('./saved_model/unet_carvana_scale0.5_epoch2.pth')
        lung_segmenter = UNet(n_channels=1, n_classes=2)
        optimizer1 = torch.optim.AdamW(lung_segmenter.parameters(), lr=1e-4)
        infection_segmenter = UNet(n_channels=1, n_classes=2)
        optimizer2 = torch.optim.AdamW(infection_segmenter.parameters(), lr=1e-4)
    elif model_name == 'pspnet':
        lung_segmenter, optimizer1 = psp_model_optimizer(layers=50) # Use default parameters
        infection_segmenter, optimizer2 = psp_model_optimizer(layers=50) # Use default parameters

    return lung_segmenter, infection_segmenter, optimizer1, optimizer2