from typing import Tuple
import torch
from torch import nn
from src.training.avg_meter import *
from src.training.metrics import *
from src.training.model_optimizer import get_model_optimizer
import os


def save_model(model, optimizer, dir) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            dir,
        )


def optimizer_to(optim, device):
    """
        Moves loaded optimizer to GPU/CPU
    """
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def model_outputs(model_name, model,
                   image_batch, mask_batch):
    main_loss, aux_loss = torch.tensor(0), torch.tensor(0)
    mask_batch = mask_batch.long()
    if model_name != 'pspnet':
        logits = model(image_batch)
        loss = model.criterion(logits, mask_batch)
    if model_name == 'esfpnet':
        y_hat = torch.sigmoid(logits)
        y_hat = (y_hat > 0.5) * 1
    elif model_name == 'unet':
        y_hat = torch.argmax(logits, dim=1)
    elif model_name == 'pspnet':
        logits, y_hat, main_loss, aux_loss = model(image_batch, mask_batch)
        loss = main_loss + 0.4 * aux_loss

    return y_hat, loss, main_loss, aux_loss


def epoch_runner(loader, lung_segmenter, infection_segmenter,
                 optimizer1, optimizer2, model_name, device):
    loss_meter = AverageMeter()
    IOU_meter = AverageMeter()
    f1_meter = AverageMeter()
    acc_meter = AverageMeter()
    for batch_number, (image, lung_mask, inf_mask) in enumerate(loader):
        n = image.shape[0]
        # Stage 1

        image = image.to(device)
        mask = lung_mask.to(device)
        
        y_hat, loss1, main_loss, aux_loss = \
            model_outputs(model_name, lung_segmenter, image, mask)

        # Stage 2

        lung_image = torch.zeros(image.shape, dtype=torch.float64).to(device) - 1000
        lung_image[y_hat.unsqueeze(1) == 1] = image[y_hat.unsqueeze(1) == 1]
        mask = inf_mask.to(device)

        y_hat, loss2, main_loss, aux_loss = \
            model_outputs(model_name, infection_segmenter, lung_image, mask)
        loss = loss1 + loss2
        accuracy, f1_score = BinaryF1(y_hat, mask)

        if optimizer1:
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

        # Clear from GPU

        for item in [image, lung_image, mask,
                     f1_score, y_hat, loss,
                     main_loss, aux_loss]:
            item = item.detach().cpu()

        loss_meter.update(val=float(loss.item()), n=n)
        iou = IOU(y_hat, mask) # Calculate IOUs
        IOU_meter.update(val=float(iou), n=n)
        f1_meter.update(val=float(f1_score), n=n)
        acc_meter.update(val=float(accuracy), n=n)

        # Empty GPU memory
        torch.cuda.empty_cache()
    return loss_meter.avg, IOU_meter.avg, f1_meter.avg, acc_meter.avg


def train(loader, lung_segmenter, infection_segmenter,
          optimizer1, optimizer2, model_name, device):
    lung_segmenter.train()
    infection_segmenter.train()

    return epoch_runner(loader, lung_segmenter, infection_segmenter,
                        optimizer1, optimizer2, model_name, device)


def validate(loader, lung_segmenter, infection_segmenter,
               model_name, device):
    lung_segmenter.eval()
    infection_segmenter.eval()

    return epoch_runner(loader, lung_segmenter, infection_segmenter,
                        None, None, model_name, device)


def predict(image, lung_mask, infection_mask, model_name, B):
    # Load models
    lung_segmenter, infection_segmenter, _, _ = \
            get_model_optimizer(model_name, B)
    lung_checkpoint = torch.load(os.path.join("saved_model",
                                   model_name,
                                   "lung_segmenter.pt"))
    infection_checkpoint = torch.load(os.path.join("saved_model",
                                        model_name,
                                        "infection_segmenter.pt"))
    
    lung_segmenter.load_state_dict(lung_checkpoint['model_state_dict'])
    infection_segmenter.load_state_dict(infection_checkpoint['model_state_dict'])

    # Make lung prediction
    y_hat, _, _, _ = model_outputs(model_name,
                                      lung_segmenter,
                                      image, lung_mask)
    ## Get lung image
    lung_image = torch.zeros(image.shape, dtype=torch.float64) - 1000
    lung_image[y_hat.unsqueeze(1) == 1] = image[y_hat.unsqueeze(1) == 1]
    
    # Make final prediction
    y_hat, _, _, _ = model_outputs(model_name,
                          infection_segmenter,
                          lung_image, infection_mask)
    
    return lung_image, y_hat