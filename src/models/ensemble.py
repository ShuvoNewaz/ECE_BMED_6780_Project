import torch

from .esfpnet import ESFPNetStructure
from .pspnet import PSPNet
from .unet_model import UNet

from scipy import signal
import numpy as np

class EnsembleNet(torch.nn.Module):
    def __init__(self, models, method):
        super().__init__()
        submodules = [self.build_submodule(i) for i in models]
        self.submodules = torch.nn.ModuleList(submodules)

        self.method = method
    def build_submodule(self, submodule):
        if submodule["model_name"] == "esfpnet":
            model = ESFPNetStructure(**submodule["args"])
        elif submodule["model_name"] == "unet":
            model = UNet(**submodule["args"])
        else: raise NotImplementedError

        state_dict = torch.load(submodule["pretrained"])
        model.load_state_dict(state_dict)
        return model
    def forward(self, x):
        logits = [submodule(x).unsqueeze(1) for submodule in self.submodules]
        logits = torch.cat(logits, dim=1) #(batch, nmodules, *logits.shape)

        if self.method == "average":
            return logits.mean(axis=1)
        else: raise NotImplementedError

class single_Conv(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, std=None, **model_args):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, 
                            kernel_size=kernel_size, **model_args)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        if std is None:
            # std = max(kernel_size) // 4 + 1
            std = 1
        kern = gkern(kernel_size, std=std) / in_channels
        kern = torch.FloatTensor(kern).repeat(1, in_channels, 1, 1)
        # kern = torch.FloatTensor(torch.ones((1, in_channels, *kernel_size)))
        self.conv.weight = torch.nn.Parameter(kern)
    def forward(self, x):
        return self.conv(x).squeeze(1)

def gkern(kernel, std=1):
    """Returns a 2D Gaussian kernel array."""
    gkern1 = signal.gaussian(kernel[0], std=std).reshape(kernel[0], 1)
    gkern2 = signal.gaussian(kernel[1], std=std).reshape(kernel[1], 1)
    gkern2d = np.outer(gkern1, gkern2)
    return gkern2d
