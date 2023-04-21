from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
import torch


def crossvalidation(dataset, k, batch_size):
    dataloader_args = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
    total_size = len(dataset)
    seg = total_size // k
    train_loader_list = []
    val_loader_list = []
    for i in range(k):
        trll = 0
        trlr = i * seg
        vall = trlr
        valr = i * seg + seg
        trrl = valr
        trrr = total_size

        train_left_indices = list(range(trll, trlr))
        train_right_indices = list(range(trrl, trrr))
        
        train_indices = train_left_indices + train_right_indices
        val_indices = list(range(vall, valr))
        
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)
        
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **dataloader_args)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, **dataloader_args)

        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
    
    return train_loader_list, val_loader_list