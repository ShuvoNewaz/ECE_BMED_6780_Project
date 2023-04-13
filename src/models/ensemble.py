import torch

from .esfpnet import ESFPNetStructure
from .pspnet import PSPNet
from .unet_model import UNet

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
