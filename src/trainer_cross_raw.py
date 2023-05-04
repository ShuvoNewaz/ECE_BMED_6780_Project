import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.train_utils_raw import *
from src.image_loader import *
from src.metrics import *
from src.avg_meter import AverageMeter, SegmentationAverageMeter
from src.crossval import crossvalidation
from src.esfpnet import ESFPNetStructure
from src.pspnet import *
from src.unet_model import UNetDummy as UNet
from typing import List, Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Subset
from typing import List


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
                    self,
                    model_name: str,
                    train_data_transforms_common: transforms.Compose,
                    train_data_transforms_image: transforms.Compose,
                    val_data_transforms: transforms.Compose,
                    train_im: str,
                    train_lung_msk: str,
                    train_inf_msk: str,
                    validation_im: str,
                    validation_lung_mask: str,
                    validation_inf_msk: str,
                    T: int=5,
                    batch_size: int=100,
                    num_classes=2,
                    k_fold=5,
                    B='B0'
                ) -> None:
        self.device = device
        self.model_dir = os.path.join('./saved_model', model_name)
        self.model_dir = self.model_dir + f'_T{T}_K{k_fold}'
        os.makedirs(self.model_dir, exist_ok=True)
        self.T = T
        self.num_classes = num_classes
        self.k_fold = k_fold
        if model_name == 'esfpnet':
            self.B = B
            self.lung_segmenter = ESFPNetStructure(self.B, 160, 0.2)
            self.optimizer_1 = torch.optim.AdamW(self.lung_segmenter.parameters(), lr=1e-4)

            self.infection_segmenter1 = []
            self.optimizer_2 = []
            for t in range(T):
                self.infection_segmenter1.append(ESFPNetStructure(self.B, 160, 0.2))
                self.optimizer_2.append(torch.optim.AdamW(self.infection_segmenter1[t].parameters(), lr=1e-4))

            self.infection_segmenter2 = ESFPNetStructure(self.B, 160, 0.2)
            self.optimizer_3 = torch.optim.AdamW(self.infection_segmenter2.parameters(), lr=1e-4)
        elif model_name == 'unet':
            pretrained_unet = torch.load('./saved_model/unet_carvana_scale0.5_epoch2.pth')
            self.lung_segmenter = UNet(n_channels=1, n_classes=2)
            self.optimizer_1 = torch.optim.AdamW(self.lung_segmenter.parameters(), lr=1e-4)

            self.infection_segmenter1 = []
            self.optimizer_2 = []
            for t in range(T):
                self.infection_segmenter1.append(UNet(n_channels=1, n_classes=2))
                self.optimizer_2.append(torch.optim.AdamW(self.infection_segmenter1[t].parameters(), lr=1e-4))
            
            self.infection_segmenter2 = UNet(n_channels=1, n_classes=num_classes)
            # self.infection_segmenter2.load_state_dict(pretrained_unet)
            self.optimizer_3 = torch.optim.AdamW(self.infection_segmenter2.parameters(), lr=1e-4)
        elif model_name == 'pspnet':
            self.lung_segmenter, self.optimizer_1 = psp_model_optimizer(layers=50) # Use default parameters

            self.infection_segmenter1 = []
            self.optimizer_2 = []
            for t in range(T):
                infection_segmenter1, optimizer_2 = psp_model_optimizer(layers=50)
                self.infection_segmenter1.append(infection_segmenter1)
                self.optimizer_2.append(optimizer_2)
            
            self.infection_segmenter2, self.optimizer_3 = psp_model_optimizer(layers=50)

        self.single_Conv = single_Conv(in_channels=T+1, kernel_size=5)
        self.single_Conv_optimizer = torch.optim.AdamW(self.single_Conv.parameters(), lr=1e-4)
                
        self.model_name = model_name
        self.best_f1 = -100
        self.best_IOU = -100
        dataloader_args = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        # dataloader_args = {}

        self.train_dataset = ImageLoader(
                                            split='train',
                                            im_file=train_im,
                                            lung_msk_file=train_lung_msk,
                                            inf_msk_file=train_inf_msk,
                                            transform_common=train_data_transforms_common,
                                            transform_image=train_data_transforms_image
                                        )
        
        self.val_dataset = ImageLoader(
                                            split='val',
                                            im_file=validation_im,
                                            lung_msk_file=validation_lung_mask,
                                            inf_msk_file=validation_inf_msk,
                                            transform_common=val_data_transforms,
                                            transform_image=None
                                        )
        
        # Set up cross-validation splits
        self.num_train_images = len(self.train_dataset)
        self.num_val_images = len(self.val_dataset)

        self.train_loader_list, self.val_loader_list = crossvalidation(self.train_dataset, k=k_fold, batch_size=batch_size)

        # Drop last batch if last batch size is 1 to keep batcnorm from breaking.

        drop_last_train = self.num_train_images % batch_size == 1
        drop_last_val = self.num_val_images % batch_size == 1
        
        self.train_loader_full = DataLoader(
                                        self.train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, **dataloader_args,
                                        drop_last=drop_last_train
                                        )
        self.val_loader_full = DataLoader(
                                        self.val_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, **dataloader_args,
                                        drop_last=drop_last_val
                                    )

        self.train_loss_history_stage1 = []
        self.validation_loss_history_stage1 = []
        self.train_IOU_history_stage1 = []
        self.validation_IOU_history_stage1 = []
        self.train_f1_history_stage1 = []
        self.validation_f1_history_stage1 = []

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_IOU_history = []
        self.validation_IOU_history = []
        self.train_f1_history = []
        self.validation_f1_history = []

        self.train_loss_history_cross = []
        self.validation_loss_history_cross = []
        self.train_IOU_history_cross = []
        self.validation_IOU_history_cross = []
        self.train_f1_history_cross = []
        self.validation_f1_history_cross = []

        for k in range(k_fold):
            self.train_loss_history_cross.append([])
            self.validation_loss_history_cross.append([])
            self.train_IOU_history_cross.append([])
            self.validation_IOU_history_cross.append([])
            self.train_f1_history_cross.append([])
            self.validation_f1_history_cross.append([])

    def run_training_loop_stage1(self, num_epochs: int, load_from_disk: bool) -> None:
        if load_from_disk:
            checkpoint = torch.load(self.model_dir+'/stage1.pt')
            self.lung_segmenter.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_1.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_to(self.optimizer_1, device)
        best_f1 = 0
        for epoch_idx in range(num_epochs):
            save_im = epoch_idx == num_epochs-1 # Save images only from the last epoch
            train_loss, train_IOU, train_f1 = self.train_stage1(save_im)

            self.train_loss_history_stage1.append(train_loss)
            self.train_IOU_history_stage1.append(train_IOU)
            self.train_f1_history_stage1.append(train_f1)

            val_loss, val_IOU, val_f1 = self.validate_stage1(save_im)

            self.validation_loss_history_stage1.append(val_loss)
            self.validation_IOU_history_stage1.append(val_IOU)
            self.validation_f1_history_stage1.append(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                save_model(self.lung_segmenter, self.optimizer_1, self.model_dir+'/stage1.pt')

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Validation Loss:{val_loss:.4f}"
                + f" Train IOU: {train_IOU:.4f}"
                + f" Validation IOU: {val_IOU:.4f}"
                + f" Train F1 Score: {train_f1:.4f}"
                + f" Validation F1 Score: {val_f1:.4f}"
                
            )

    def run_training_loop_stage2_3(self, num_epochs: int, load_from_disk: bool) -> None:
        if load_from_disk:
            for t in range(self.T):
                # Stage 2
                checkpoint = torch.load(self.model_dir+f'/stage2_{t+1}_final.pt')
                self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_to(self.optimizer_2[t], device)
            # Stage 3
            checkpoint = torch.load(self.model_dir+'/stage3_final.pt')
            self.infection_segmenter2.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_to(self.optimizer_3, device)
        else: # load the best cross-validation parameters
            for t in range(self.T):
                # Stage 2
                checkpoint = torch.load(self.model_dir+f'/stage2_{t+1}.pt')
                self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_to(self.optimizer_2[t], device)
            # Stage 3
            checkpoint = torch.load(self.model_dir+'/stage3.pt')
            self.infection_segmenter2.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_to(self.optimizer_3, device)
        for epoch_idx in range(num_epochs):
            save_im = epoch_idx == num_epochs-1 # Save images only from the last epoch

            train_loss, train_IOU, train_f1 = self.train_stage2_3(save_im)
            self.train_loss_history.append(train_loss)
            self.train_IOU_history.append(train_IOU)
            self.train_f1_history.append(train_f1)

            val_loss, val_IOU, val_f1 = self.validate_stage2_3(save_im)
            self.validation_loss_history.append(val_loss)
            self.validation_IOU_history.append(val_IOU)
            self.validation_f1_history.append(val_f1)

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train IOU: {train_IOU:.4f}"
                + f" Validation IOU: {val_IOU:.4f}"
                + f" Train F1 Score: {train_f1:.4f}"
                + f" Validation F1 Score: {val_f1:.4f}"
            )
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.best_IOU = val_IOU

                for t in range(self.T):
                    save_model(self.infection_segmenter1[t], self.optimizer_2[t], self.model_dir+f'/stage2_{t+1}_final.pt')
                save_model(self.infection_segmenter2, self.optimizer_3, self.model_dir+'/stage3_final.pt')

    def train_stage1(self, save_im: bool) -> Tuple[float, float]:
        # Set training mode
        
        self.lung_segmenter.train()

        train_loss_meter = AverageMeter()
        train_IOU_meter = AverageMeter()
        train_f1_meter = AverageMeter()

        # loop over each minibatch

        for batch_number, (image, lung_image, lung_mask, inf_mask) in enumerate(self.train_loader_full):
            image = image.to(device)
            lung_mask = lung_mask.to(device)
            n = image.shape[0]

            # Stage 1

            lung_mask = lung_mask.long()
            self.lung_segmenter = self.lung_segmenter.to(device)
            
            if self.model_name != 'pspnet':
                lung_logits = self.lung_segmenter(image)
                stage1_loss = self.lung_segmenter.criterion(lung_logits, lung_mask)
            if self.model_name == 'esfpnet':
                y_hat = torch.sigmoid(lung_logits)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'unet':
                y_hat = torch.argmax(lung_logits, dim=1)
            elif self.model_name == 'pspnet':
                lung_logits, y_hat, main_loss, aux_loss = self.lung_segmenter(image, lung_mask)
                stage1_loss = main_loss + 0.4 * aux_loss
            f1_score = BinaryF1(y_hat, lung_mask)
            self.optimizer_1.zero_grad()
            stage1_loss.backward()
            self.optimizer_1.step()

            # Clear from GPU

            self.lung_segmenter = self.lung_segmenter.cpu()
            image = image.detach().cpu()
            lung_logits = lung_logits.detach().cpu()
            lung_mask = lung_mask.detach().cpu()
            stage1_loss = stage1_loss.detach().cpu()
            y_hat = y_hat.detach().cpu()
            f1_score = f1_score.detach().cpu()
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()

            train_loss_meter.update(val=float(stage1_loss.item()), n=n)

            iou = IOU(y_hat, lung_mask) # Calculate IOUs
            train_IOU_meter.update(val=float(iou), n=n)
            train_f1_meter.update(val=float(f1_score), n=n)

            # Empty GPU memory

            torch.cuda.empty_cache()

            # print('After clearing', torch.cuda.memory_allocated(0))

            if save_im:
                if batch_number == 0:
                    self.original_images_train = image
                    self.lung_images_train = lung_image
                    self.lung_mask_train = lung_mask
                    self.original_masks_train = inf_mask
                else:
                    self.original_images_train = np.concatenate((self.original_images_train, image))
                    self.lung_images_train = np.concatenate((self.lung_images_train, lung_image))
                    self.lung_mask_train = np.concatenate((self.lung_mask_train, lung_mask))
                    self.original_masks_train = np.concatenate((self.original_masks_train, inf_mask))

        return train_loss_meter.avg, train_IOU_meter.avg, train_f1_meter.avg
    
    def validate_stage1(self, save_im: bool=False) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        # Set validation mode
        
        self.lung_segmenter.eval()

        val_loss_meter = AverageMeter()
        val_IOU_meter = AverageMeter()
        val_f1_meter = AverageMeter()

        # loop over each minibatch

        for batch_number, (image, lung_image, lung_mask, inf_mask) in enumerate(self.val_loader_full):
            image = image.to(device)
            lung_mask = lung_mask.to(device)
            n = image.shape[0]

            # Stage 1

            lung_mask = lung_mask.long()
            self.lung_segmenter = self.lung_segmenter.to(device)

            if self.model_name != 'pspnet':
                lung_logits = self.lung_segmenter(image)
                stage1_loss = self.lung_segmenter.criterion(lung_logits, lung_mask)
            if self.model_name == 'esfpnet':
                y_hat = torch.sigmoid(lung_logits)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'unet':
                y_hat = torch.argmax(lung_logits, dim=1)
            elif self.model_name == 'pspnet':
                lung_logits, y_hat, main_loss, aux_loss = self.lung_segmenter(image, lung_mask)
                stage1_loss = main_loss + 0.4 * aux_loss
            f1_score = BinaryF1(y_hat, lung_mask)
            
            # Clear from GPU

            self.lung_segmenter = self.lung_segmenter.cpu()
            image = image.detach().cpu()
            lung_logits = lung_logits.detach().cpu()
            lung_mask = lung_mask.detach().cpu()
            stage1_loss = stage1_loss.detach().cpu()
            y_hat = y_hat.detach().cpu()
            f1_score = f1_score.detach().cpu()
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()

            val_loss_meter.update(val=float(stage1_loss.item()), n=n)

            iou = IOU(y_hat, lung_mask) # Calculate IOUs
            val_IOU_meter.update(val=float(iou), n=n)
            val_f1_meter.update(val=float(f1_score), n=n)

            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            stage1_loss = stage1_loss.detach().cpu()

            # Empty GPU memory

            torch.cuda.empty_cache()

            if save_im:
                if batch_number == 0:
                    self.original_images_val = image
                    self.original_lung_images_val = lung_image
                    self.predicted_lung_mask_val = y_hat
                    self.lung_mask_val = lung_mask
                    self.original_masks_val = inf_mask
                else:
                    self.original_images_val = np.concatenate((self.original_images_val, image))
                    self.original_lung_images_val = np.concatenate((self.original_lung_images_val, lung_image))
                    self.predicted_lung_mask_val = np.concatenate((self.predicted_lung_mask_val, y_hat))
                    self.lung_mask_val = np.concatenate((self.lung_mask_val, lung_mask))
                    self.original_masks_val = np.concatenate((self.original_masks_val, inf_mask))

        return val_loss_meter.avg, val_IOU_meter.avg, val_f1_meter.avg

    def crossval_epoch(self, num_epochs: int, load_from_disk: bool):
        best_f1 = -100
        best_IOU = -100
        print(f'Starting {self.k_fold}-fold cross-validation...\n')
        for fold_number, (train_loader, val_loader) in enumerate(zip(self.train_loader_list, self.val_loader_list)): 
            # Choose whether to load model and optimizer from disk
            if load_from_disk:
                for t in range(self.T):
                    # Stage 2
                    checkpoint = torch.load(self.model_dir+f'/stage2_{t+1}_final.pt')
                    self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
                    optimizer_to(self.optimizer_2[t], device)
                # Stage 3
                checkpoint = torch.load(self.model_dir+'/stage3_final.pt')
                self.infection_segmenter2.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_to(self.optimizer_3, device)
            # Run over the epochs
            for epoch in range(num_epochs):
                train_loss, train_IOU, train_f1 = TrainCross(train_loader, self.model_name, self.infection_segmenter1, self.infection_segmenter2,
                                                                self.optimizer_2, self.optimizer_3, self.T, self.num_classes, self.single_Conv, self.single_Conv_optimizer)
                self.train_loss_history_cross[fold_number].append(train_loss)
                self.train_IOU_history_cross[fold_number].append(train_IOU)
                self.train_f1_history_cross[fold_number].append(train_f1)

                val_loss, val_IOU, val_f1 = ValidateCross(val_loader, self.model_name, self.infection_segmenter1, self.infection_segmenter2,
                                                            self.T, self.num_classes, self.single_Conv)
                self.validation_loss_history_cross[fold_number].append(val_loss)
                self.validation_IOU_history_cross[fold_number].append(val_IOU)
                self.validation_f1_history_cross[fold_number].append(val_f1)

                # Save best model

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_IOU = val_IOU
                    for t in range(self.T):
                        save_model(self.infection_segmenter1[t], self.optimizer_2[t], self.model_dir+f'/stage2_{t+1}.pt')
                    save_model(self.infection_segmenter2, self.optimizer_3, self.model_dir+'/stage3.pt')
            print(f'Fold {fold_number+1}/{self.k_fold} completed!')
        
        return best_f1, best_IOU

    def train_stage2_3(self, save_im: bool) -> Tuple[float, float]:
        # Set training mode
        
        for t in range(self.T):
            self.infection_segmenter1[t].train()
        self.infection_segmenter2.train()

        train_loss_meter = AverageMeter()
        train_IOU_meter = AverageMeter()
        train_f1_meter = AverageMeter()

        # loop over each minibatch

        for batch_number, (image, lung_image, lung_mask, inf_mask) in enumerate(self.train_loader_full):
            image = image.to(device) # LUNG IMAGE IS NOT LUNG IMAGE
            inf_mask = inf_mask.to(device)
            n = image.shape[0]

            lung_mask = lung_mask.long()
            inf_mask = inf_mask.long()

            # Stage 2
            
            T_probabilities = torch.zeros(self.T, n, self.num_classes, image.shape[2], image.shape[3])
            stage2_loss_total = 0

            for t in range(self.T):
                self.infection_segmenter1[t] = self.infection_segmenter1[t].to(device)
                if self.model_name != 'pspnet':
                    inf_logits1 = self.infection_segmenter1[t](image)
                    stage2_loss = self.infection_segmenter1[t].criterion(inf_logits1, inf_mask)
                if self.model_name == 'esfpnet':
                    y_hat_inf_1 = torch.sigmoid(inf_logits1)
                    prob = torch.concat((torch.unsqueeze(y_hat_inf_1, dim=1), 1-torch.unsqueeze(y_hat_inf_1, dim=1)), dim=1)
                    y_hat_inf_1 = (y_hat_inf_1 > 0.5) * 1
                elif self.model_name == 'unet':
                    y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)
                    prob = nn.Softmax(dim=1)(inf_logits1)
                elif self.model_name == 'pspnet':
                    inf_logits1, y_hat_inf_1, main_loss, aux_loss = self.infection_segmenter1[t](image, inf_mask)
                    stage2_loss = main_loss + 0.4 * aux_loss
                    prob = nn.Softmax(dim=1)(inf_logits1)

                self.optimizer_2[t].zero_grad()
                stage2_loss.backward()
                self.optimizer_2[t].step()

                if t == 0:
                    T_preds = torch.unsqueeze(y_hat_inf_1, dim=1)
                else:
                    T_preds = torch.concat((T_preds, torch.unsqueeze(y_hat_inf_1, dim=1)), dim=1)

                # Clear from GPU

                prob = prob.detach().cpu()
                self.infection_segmenter1[t] = self.infection_segmenter1[t].cpu()
                inf_logits1 = inf_logits1.detach().cpu()
                stage2_loss = stage2_loss.detach().cpu()
                stage2_loss_total = stage2_loss_total + stage2_loss
                T_probabilities[t] = prob # Store T probabilities for each image in the batch

            # self.single_Conv = self.single_Conv.to(device)

            T_probabilities = T_probabilities.detach()
            T_preds = T_preds.detach()
            T_preds = torch.mean(T_preds.float(), dim=1, keepdim=True)
            sam_var = sample_variance(T_probabilities)
            pred_ent = predictive_entropy(T_probabilities)
            sam_var = sam_var.to(device)
            pred_ent = pred_ent.to(device)

            # T_preds = self.single_Conv(T_preds)
            # T_preds = self.single_Conv(torch.concat((image, T_preds), dim=1))
            # conv_loss = self.single_Conv.criterion(T_preds, inf_mask.float())
            # self.single_Conv_optimizer.zero_grad()
            # conv_loss.backward()
            # self.single_Conv_optimizer.step()
            # T_preds = T_preds.detach()
            # conv_loss = conv_loss.detach().cpu()
            # self.single_Conv = self.single_Conv.cpu()
            # T_preds = torch.unsqueeze(T_preds, dim=1)
            
            # Stage 3

            self.infection_segmenter2 = self.infection_segmenter2.to(device)
            if self.model_name != 'pspnet':
                inf_logits2 = self.infection_segmenter2(torch.concat((image, T_preds, torch.unsqueeze(pred_ent, dim=1)), dim=1))
                # inf_logits2 = self.infection_segmenter2(T_preds)
                stage3_loss = self.infection_segmenter2.criterion(inf_logits2, inf_mask)
            if self.model_name == 'esfpnet':
                y_hat = torch.sigmoid(inf_logits2)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'unet':
                y_hat = torch.argmax(inf_logits2, dim=1)
            elif self.model_name == 'pspnet':
                inf_logits2, y_hat, main_loss, aux_loss = self.infection_segmenter2(torch.concat((image, T_preds, torch.unsqueeze(pred_ent, dim=1)), dim=1), inf_mask)
                # inf_logits2, y_hat, main_loss, aux_loss = self.infection_segmenter2(T_preds, inf_mask)
                stage3_loss = main_loss + 0.4 * aux_loss

            f1_score = BinaryF1(y_hat, inf_mask)

            self.optimizer_3.zero_grad()
            stage3_loss.backward()
            self.optimizer_3.step()

            # Clear from GPU

            self.infection_segmenter2 = self.infection_segmenter2.cpu()
            y_hat_inf_1 = y_hat_inf_1.detach().cpu()
            inf_logits2 = inf_logits2.detach().cpu()
            stage3_loss = stage3_loss.detach().cpu()
            f1_score = f1_score.detach().cpu()
            image = image.detach().cpu()
            train_f1_meter.update(val=float(f1_score.item()), n=n)

            batch_loss = stage2_loss_total + stage3_loss# + conv_loss
            train_loss_meter.update(val=float(batch_loss.item()), n=n)
            
            # Empty GPU memory

            inf_mask = inf_mask.detach().cpu()
            y_hat = y_hat.detach().cpu()
            T_preds = T_preds.cpu()
            T_probabilities = T_probabilities.cpu()
            sam_var = sam_var.detach().cpu()
            pred_ent = pred_ent.detach().cpu()

            iou = IOU(y_hat, inf_mask) # Calculate IOUs
            train_IOU_meter.update(val=float(iou), n=n)
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

            if save_im:
                if batch_number == 0:
                    self.predictions_train = y_hat
                    if self.model_name == 'unet':
                        self.probability_train = prob
                else:
                    self.predictions_train = np.concatenate((self.predictions_train, y_hat))
                    if self.model_name == 'unet':
                        self.probability_train = np.concatenate((self.probability_train, prob))

        return train_loss_meter.avg, train_IOU_meter.avg, train_f1_meter.avg

    def validate_stage2_3(self, save_im: bool) -> Tuple[float, float]:
        # Set validation mode
        
        for t in range(self.T):
            self.infection_segmenter1[t].eval()
        self.infection_segmenter2.eval()

        val_loss_meter = AverageMeter()
        val_IOU_meter = AverageMeter()
        val_f1_meter = AverageMeter()

        # loop over each minibatch

        index_count = 0
        for batch_number, (image, lung_image, lung_mask, inf_mask) in enumerate(self.val_loader_full):
            image = image.to(device)
            inf_mask = inf_mask.to(device)
            n = image.shape[0]

            # Stage 2
            
            inf_mask = inf_mask.long()                
            T_probabilities = torch.zeros(self.T, n, self.num_classes, image.shape[2], image.shape[3])
            # T_preds = torch.zeros(n, self.T, image.shape[2], image.shape[3])
            stage2_loss_total = 0

            for t in range(self.T):
                self.infection_segmenter1[t] = self.infection_segmenter1[t].to(device)
                if self.model_name != 'pspnet':
                    inf_logits1 = self.infection_segmenter1[t](image)
                    stage2_loss = self.infection_segmenter1[t].criterion(inf_logits1, inf_mask)
                if self.model_name == 'esfpnet':
                    y_hat_inf_1 = torch.sigmoid(inf_logits1)
                    prob = torch.concat((torch.unsqueeze(y_hat_inf_1, dim=1), 1-torch.unsqueeze(y_hat_inf_1, dim=1)), dim=1)
                    y_hat_inf_1 = (y_hat_inf_1 > 0.5) * 1
                elif self.model_name == 'unet':
                    y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)
                    prob = nn.Softmax(dim=1)(inf_logits1)
                elif self.model_name == 'pspnet':
                    inf_logits1, y_hat_inf_1, main_loss, aux_loss = self.infection_segmenter1[t](image, inf_mask)
                    stage2_loss = main_loss + 0.4 * aux_loss
                    prob = nn.Softmax(dim=1)(inf_logits1)

                if t == 0:
                    T_preds = torch.unsqueeze(y_hat_inf_1, dim=1)
                else:
                    T_preds = torch.concat((T_preds, torch.unsqueeze(y_hat_inf_1, dim=1)), dim=1)
                
                # Clear from GPU

                prob = prob.detach().cpu()
                self.infection_segmenter1[t] = self.infection_segmenter1[t].cpu()
                inf_logits1 = inf_logits1.detach().cpu()
                stage2_loss = stage2_loss.detach().cpu()
                stage2_loss_total = stage2_loss_total + stage2_loss
                T_probabilities[t] = prob # Store T probabilities for each image in the batch

            # self.single_Conv = self.single_Conv.to(device)

            T_preds = T_preds.detach()
            T_preds = torch.mean(T_preds.float(), dim=1, keepdim=True)
            T_probabilities = T_probabilities.detach()
            sam_var = sample_variance(T_probabilities)
            pred_ent = predictive_entropy(T_probabilities)
            sam_var = sam_var.to(device)
            pred_ent = pred_ent.to(device)

            # T_preds = self.single_Conv(T_preds)
            # T_preds = self.single_Conv(torch.concat((image, T_preds), dim=1))
            # conv_loss = self.single_Conv.criterion(T_preds, inf_mask.float())
            # self.single_Conv_optimizer.zero_grad()
            # conv_loss.backward()
            # self.single_Conv_optimizer.step()
            # T_preds = T_preds.detach()
            # conv_loss = conv_loss.detach().cpu()
            # self.single_Conv = self.single_Conv.cpu()
            # T_preds = torch.unsqueeze(T_preds, dim=1)
            
            # Stage 3

            self.infection_segmenter2 = self.infection_segmenter2.to(device)
            if self.model_name != 'pspnet':
                inf_logits2 = self.infection_segmenter2(torch.concat((image, T_preds, torch.unsqueeze(pred_ent, dim=1)), dim=1))
                # inf_logits2 = self.infection_segmenter2(T_preds)
                stage3_loss = self.infection_segmenter2.criterion(inf_logits2, inf_mask)
            if self.model_name == 'esfpnet':
                y_hat = torch.sigmoid(inf_logits2)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'unet':
                y_hat = torch.argmax(inf_logits2, dim=1)
            elif self.model_name == 'pspnet':
                inf_logits2, y_hat, main_loss, aux_loss = self.infection_segmenter2(torch.concat((image, T_preds, torch.unsqueeze(pred_ent, dim=1)), dim=1), inf_mask)
                # inf_logits2, y_hat, main_loss, aux_loss = self.infection_segmenter2(T_preds, inf_mask)
                stage3_loss = main_loss + 0.4 * aux_loss

            f1_score = BinaryF1(y_hat, inf_mask)

            # Clear from GPU

            self.infection_segmenter2 = self.infection_segmenter2.cpu()
            inf_logits2 = inf_logits2.detach().cpu()
            stage3_loss = stage3_loss.detach().cpu()
            f1_score = f1_score.detach().cpu()
            image = image.detach().cpu()

            batch_loss = stage2_loss_total + stage3_loss# + conv_loss

            val_loss_meter.update(val=float(batch_loss.item()), n=n)
            val_f1_meter.update(val=float(f1_score.item()), n=n)

            # Empty GPU memory

            inf_mask = inf_mask.detach().cpu()
            y_hat_inf_1 = y_hat_inf_1.detach().cpu()
            y_hat = y_hat.detach().cpu()
            T_preds = T_preds.cpu()
            T_probabilities = T_probabilities.cpu()
            sam_var = sam_var.detach().cpu()
            pred_ent = pred_ent.detach().cpu()

            iou = IOU(y_hat, inf_mask) # Calculate IOUs
            val_IOU_meter.update(val=float(iou), n=n)
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

            if save_im:
                if batch_number == 0:
                    self.var = sam_var
                    self.pred_ent = pred_ent
                    self.original_images_val = image
                    self.lung_images_val = lung_image
                    self.original_masks_val = inf_mask
                    self.predictions_val = y_hat
                    if self.model_name == 'unet':
                        self.probability_val = prob
                else:
                    self.var = np.concatenate((self.var, sam_var))
                    self.pred_ent = np.concatenate((self.pred_ent, pred_ent))
                    self.original_images_val = np.concatenate((self.original_images_val, image))
                    self.lung_images_val = np.concatenate((self.lung_images_val, lung_image))
                    self.original_masks_val = np.concatenate((self.original_masks_val, inf_mask))
                    self.predictions_val = np.concatenate((self.predictions_val, y_hat))
                    if self.model_name == 'unet':
                        self.probability_val = np.concatenate((self.probability_val, prob))

        return val_loss_meter.avg, val_IOU_meter.avg, val_f1_meter.avg
    

    def Predict(self, image, load_model_from_disk: bool):
        assert image.ndim == 4
        if load_model_from_disk:
            # Stage 1
            checkpoint = torch.load(self.model_dir+'/stage1.pt')
            self.lung_segmenter.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_1.load_state_dict(checkpoint['optimizer_state_dict'])
            for t in range(self.T):
                # Stage 2
                checkpoint = torch.load(self.model_dir+f'/stage2_{t+1}_final.pt')
                self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
            # Stage 3
            checkpoint = torch.load(self.model_dir+'/stage3_final.pt')
            self.infection_segmenter2.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
        
        image = image.to(device)
        n = image.shape[0]

        # Stage 1

        self.lung_segmenter = self.lung_segmenter.to(device)

        if self.model_name != 'pspnet':
            lung_logits = self.lung_segmenter(image)
        if self.model_name == 'esfpnet':
            lung_mask = torch.sigmoid(lung_logits)
            lung_mask = (lung_mask > 0.5) * 1
        elif self.model_name == 'unet':
            lung_mask = torch.argmax(lung_logits, dim=1)
        elif self.model_name == 'pspnet':
            lung_logits, lung_mask, main_loss, aux_loss = self.lung_segmenter(image)

        # Create the lung regions

        lung_image = torch.zeros(image.shape).float().to(device)
        lung_index = lung_mask[0] != 0
        background_index = lung_mask[0] == 0
        lung_image[0, 0][lung_index] = image[0, 0][lung_index]
        lung_image[0, 0][background_index] = torch.min(image[0, 0])

        # Clear from GPU

        self.lung_segmenter = self.lung_segmenter.cpu()
        lung_logits = lung_logits.detach().cpu()
        lung_mask = lung_mask.detach().cpu()
        lung_image = lung_image.detach().cpu()

        # Stage 2

        T_probabilities = torch.zeros(self.T, n, self.num_classes, image.shape[2], image.shape[3])
        # T_preds = torch.zeros(n, self.T, image.shape[2], image.shape[3])
        stage2_loss_total = 0

        for t in range(self.T):
            self.infection_segmenter1[t] = self.infection_segmenter1[t].to(device)
            if self.model_name != 'pspnet':
                inf_logits1 = self.infection_segmenter1[t](image)
            if self.model_name == 'esfpnet':
                y_hat_inf_1 = torch.sigmoid(inf_logits1)
                prob = torch.concat((torch.unsqueeze(y_hat_inf_1, dim=1), 1-torch.unsqueeze(y_hat_inf_1, dim=1)), dim=1)
                y_hat_inf_1 = (y_hat_inf_1 > 0.5) * 1
            elif self.model_name == 'unet':
                y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)
                prob = nn.Softmax(dim=1)(inf_logits1)
            elif self.model_name == 'pspnet':
                inf_logits1, y_hat_inf_1, main_loss, aux_loss = self.infection_segmenter1[t](image)
                prob = nn.Softmax(dim=1)(inf_logits1)

            if t == 0:
                T_preds = torch.unsqueeze(y_hat_inf_1, dim=1)
            else:
                T_preds = torch.concat((T_preds, torch.unsqueeze(y_hat_inf_1, dim=1)), dim=1)
            
            # Clear from GPU

            prob = prob.detach().cpu()
            self.infection_segmenter1[t] = self.infection_segmenter1[t].cpu()
            inf_logits1 = inf_logits1.detach().cpu()
            T_probabilities[t] = prob # Store T probabilities for each image in the batch

        T_preds = T_preds.detach()
        T_preds = torch.mean(T_preds.float(), dim=1, keepdim=True)
        T_probabilities = T_probabilities.detach()
        sam_var = sample_variance(T_probabilities)
        pred_ent = predictive_entropy(T_probabilities)
        sam_var = sam_var.to(device)
        pred_ent = pred_ent.to(device)

        # Stage 3

        self.infection_segmenter2 = self.infection_segmenter2.to(device)
        if self.model_name != 'pspnet':
            inf_logits2 = self.infection_segmenter2(torch.concat((image, T_preds, torch.unsqueeze(pred_ent, dim=1)), dim=1))
        if self.model_name == 'esfpnet':
            y_hat = torch.sigmoid(inf_logits2)
            y_hat = (y_hat > 0.5) * 1
        elif self.model_name == 'unet':
            y_hat = torch.argmax(inf_logits2, dim=1)
        elif self.model_name == 'pspnet':
            inf_logits2, y_hat, main_loss, aux_loss = self.infection_segmenter2(torch.concat((image, T_preds, torch.unsqueeze(pred_ent, dim=1)), dim=1))

        # Clear from GPU

        self.infection_segmenter2 = self.infection_segmenter2.cpu()
        inf_logits2 = inf_logits2.detach().cpu()
        image = image.detach().cpu()

        # Empty GPU memory

        y_hat_inf_1 = y_hat_inf_1.detach().cpu()
        y_hat = y_hat.detach().cpu()
        T_preds = T_preds.cpu()
        T_probabilities = T_probabilities.cpu()
        sam_var = sam_var.detach().cpu()
        pred_ent = pred_ent.detach().cpu()
        torch.cuda.empty_cache()

        return lung_mask, lung_image, y_hat
