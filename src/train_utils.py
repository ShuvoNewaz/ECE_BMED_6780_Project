import torch
from src.metrics import *
from src.avg_meter import *
from torch import nn
import matplotlib.pyplot as plt
from typing import List


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def TrainCross(train_loader, model_name, infection_segmenter1: List[nn.Module],
               infection_segmenter2: List[nn.Module], optimizer_2, optimizer_3, T, num_classes):
    
    for t in range(T):
        infection_segmenter1[t].train()
    infection_segmenter2.train()

    train_loss_meter = AverageMeter()
    train_IOU_meter = AverageMeter()
    train_f1_meter = AverageMeter()
    for batch_number, (image, lung_image, lung_mask, inf_mask) in enumerate(train_loader):
        # print('Before everything to GPU', torch.cuda.memory_allocated(0))

        lung_image = lung_image.to(device)
        inf_mask = inf_mask.to(device)

        # print('After Data to GPU', torch.cuda.memory_allocated(0))

        n = image.shape[0]
        if model_name == 'esfpnet':
            logits = self.model(x)
            batch_loss = ange_structure_loss(logits, masks)
            y_hat = torch.sigmoid(logits)
            y_hat = (y_hat > 0.5) * 1
        elif model_name == 'pspnet':
            logits, y_hat, aux_loss, main_loss = self.model(x, masks)
            aux_weight = 0.4
            batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
        elif model_name == 'unet':
            lung_mask = lung_mask.long()
            inf_mask = inf_mask.long()

            # Stage 2
            
            T_probabilities = torch.zeros(T, n, num_classes, image.shape[2], image.shape[3])
            # T_preds = torch.zeros(n, self.T, image.shape[2], image.shape[3])
            stage2_loss_total = 0
            for t in range(T):
                infection_segmenter1[t] = infection_segmenter1[t].to(device)
                inf_logits1 = infection_segmenter1[t](lung_image)
                prob = nn.Softmax(dim=1)(inf_logits1)
                stage2_loss = infection_segmenter1[t].criterion(inf_logits1, inf_mask)
                y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)

                optimizer_2[t].zero_grad()
                stage2_loss.backward()
                optimizer_2[t].step()

                # Clear from GPU

                prob = prob.detach().cpu()
                infection_segmenter1[t] = infection_segmenter1[t].cpu()
                inf_logits1 = inf_logits1.detach().cpu()
                y_hat_inf_1 = y_hat_inf_1.detach().cpu()
                stage2_loss = stage2_loss.detach().cpu()
                stage2_loss_total = stage2_loss_total + stage2_loss
                T_probabilities[t] = prob # Store T probabilities for each image in the batch
                # T_preds[:, t] = y_hat_inf_1

            # T_preds = T_preds.detach()
            T_probabilities = T_probabilities.detach()
            sam_var = sample_variance(T_probabilities)
            pred_ent = predictive_entropy(T_probabilities)
            sam_var = sam_var.to(device)
            pred_ent = pred_ent.to(device)

            # Stage 3

            infection_segmenter2 = infection_segmenter2.to(device)
            inf_logits2 = infection_segmenter2(torch.concat((lung_image, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
            stage3_loss = infection_segmenter2.criterion(inf_logits2, inf_mask)
            y_hat = torch.argmax(inf_logits2, dim=1)
            f1_score = BinaryF1(y_hat, inf_mask)

            optimizer_3.zero_grad()
            stage3_loss.backward()
            optimizer_3.step()

            # Clear from GPU

            infection_segmenter2 = infection_segmenter2.cpu()
            inf_logits2 = inf_logits2.detach().cpu()
            stage3_loss = stage3_loss.detach().cpu()
            f1_score = f1_score.detach().cpu()
            lung_image = lung_image.detach().cpu()
            # print('After stage 3 to CPU', torch.cuda.memory_allocated(0))
            train_f1_meter.update(val=float(f1_score.item()), n=n)

            batch_loss = stage2_loss_total + stage3_loss


        train_loss_meter.update(val=float(batch_loss.item()), n=n)
        
        # Empty GPU memory

        inf_mask = inf_mask.detach().cpu()
        y_hat = y_hat.detach().cpu()
        # T_preds = T_preds.cpu()
        T_probabilities = T_probabilities.cpu()
        sam_var = sam_var.detach().cpu()
        pred_ent = pred_ent.detach().cpu()

        iou = IOU(y_hat, inf_mask) # Calculate IOUs
        train_IOU_meter.update(val=float(iou), n=n)
        if model_name == 'pspnet':
            main_loss = main_loss.detach().cpu()
            aux_loss = aux_loss.detach().cpu()
        batch_loss = batch_loss.detach().cpu()
        torch.cuda.empty_cache()

        # print('After clearing', torch.cuda.memory_allocated(0))
    return train_loss_meter.avg, train_IOU_meter.avg, train_f1_meter.avg


def ValidateCross(val_loader, model_name, infection_segmenter1, infection_segmenter2,
                        T, num_classes):
    """
        Implements the validation pass in cross-validation. Unlike external validation
        set, this set has ground truth lung masks because it extracted from the training set.
    """
    # Set validation mode
        
    for t in range(T):
        infection_segmenter1[t].eval()
    infection_segmenter2.eval()

    val_loss_meter = AverageMeter()
    val_IOU_meter = AverageMeter()
    val_f1_meter = AverageMeter()

    for batch_number, (image, lung_image, lung_mask, inf_mask) in enumerate(val_loader):
        image = image.to(device)
        lung_image = lung_image.to(device)
        inf_mask = inf_mask.to(device)

        n = image.shape[0]

        # inf_mask = torch.squeeze(inf_mask, dim=1) # Compensating for the transforms
        
        if model_name == 'esfpnet':
            logits = self.model(x)
            batch_loss = ange_structure_loss(logits, masks)
            y_hat = torch.sigmoid(logits)
            y_hat = (y_hat > 0.5) * 1
        elif model_name == 'pspnet':
            logits, y_hat, aux_loss, main_loss = self.model(x, masks)
            aux_weight = 0.4
            batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
        elif model_name == 'unet':
            inf_mask = inf_mask.long()

            # Stage 2
            
            T_probabilities = torch.zeros(T, n, num_classes, image.shape[2], image.shape[3])
            # T_preds = torch.zeros(n, self.T, image.shape[2], image.shape[3])
            stage2_loss_total = 0
            for t in range(T):
                infection_segmenter1[t] = infection_segmenter1[t].to(device)
                inf_logits1 = infection_segmenter1[t](lung_image)
                prob = nn.Softmax(dim=1)(inf_logits1)
                stage2_loss = infection_segmenter1[t].criterion(inf_logits1, inf_mask)
                y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)
                
                # Clear from GPU

                prob = prob.detach().cpu()
                infection_segmenter1[t] = infection_segmenter1[t].cpu()
                inf_logits1 = inf_logits1.detach().cpu()
                y_hat_inf_1 = y_hat_inf_1.detach().cpu()
                stage2_loss = stage2_loss.detach().cpu()
                stage2_loss_total = stage2_loss_total + stage2_loss
                T_probabilities[t] = prob # Store T probabilities for each image in the batch
                # T_preds[:, t] = y_hat_inf_1

            # T_preds = T_preds.detach()
            T_probabilities = T_probabilities.detach()
            sam_var = sample_variance(T_probabilities)
            pred_ent = predictive_entropy(T_probabilities)
            sam_var = sam_var.to(device)
            pred_ent = pred_ent.to(device)

            # T_probabilities = torch.swapaxes(T_probabilities, 0, 1)
            # T_probabilities = T_probabilities.reshape(n, self.T*self.num_classes, image.shape[2], image.shape[3])
            
            # Stage 3

            infection_segmenter2 = infection_segmenter2.to(device)
            # inf_logits2 = self.infection_segmenter2(torch.concat((T_preds, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
            # inf_logits2 = self.infection_segmenter2(torch.concat((T_probabilities, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
            inf_logits2 = infection_segmenter2(torch.concat((lung_image, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
            stage3_loss = infection_segmenter2.criterion(inf_logits2, inf_mask)
            y_hat = torch.argmax(inf_logits2, dim=1)
            f1_score = BinaryF1(y_hat, inf_mask)

            # Clear from GPU

            infection_segmenter2 = infection_segmenter2.cpu()
            inf_logits2 = inf_logits2.detach().cpu()
            stage3_loss = stage3_loss.detach().cpu()
            f1_score = f1_score.detach().cpu()
            lung_image = lung_image.detach().cpu()

            batch_loss = stage2_loss_total + stage3_loss

            val_loss_meter.update(val=float(batch_loss.item()), n=n)
            val_f1_meter.update(val=float(f1_score.item()), n=n)

        # Empty GPU memory

        inf_mask = inf_mask.detach().cpu()
        y_hat = y_hat.detach().cpu()
        # T_preds = T_preds.cpu()
        T_probabilities = T_probabilities.cpu()
        sam_var = sam_var.detach().cpu()
        pred_ent = pred_ent.detach().cpu()

        iou = IOU(y_hat, inf_mask) # Calculate IOUs
        val_IOU_meter.update(val=float(iou), n=n)
        if model_name == 'pspnet':
            main_loss = main_loss.detach().cpu()
            aux_loss = aux_loss.detach().cpu()
        batch_loss = batch_loss.detach().cpu()
        torch.cuda.empty_cache()

    return val_loss_meter.avg, val_IOU_meter.avg, val_f1_meter.avg


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


def plot_history(plot_description: str, plot_list_train: List, plot_list_val: List=None):
    epoch_idxs = range(len(plot_list_train))
    plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
    plt.plot(epoch_idxs, plot_list_train, "-b", label="training")
    if plot_list_val:
        plt.plot(epoch_idxs, plot_list_val, "-r", label="validation")
    plt.title(plot_description)
    plt.legend()
    plt.ylabel(plot_description.split(' ')[0])
    plt.xlabel("Epochs")