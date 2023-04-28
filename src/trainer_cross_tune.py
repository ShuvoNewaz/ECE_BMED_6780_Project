import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.train_utils import *
from src.image_loader import *
from src.metrics import *
from src.avg_meter import AverageMeter, SegmentationAverageMeter
from src.crossval import crossvalidation
# from src.esfpnet import ESFPNetStructure
from src.pspnet import *
from src.unet_model import UNet#Dummy as UNet
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
                    model_dir: str,
                    train_data_transforms: transforms.Compose,
                    val_data_transforms: transforms.Compose,
                    train_im: str,
                    train_lung_msk: str,
                    train_inf_msk: str,
                    validation_im: str,
                    validation_inf_msk: str,
                    lr_list_stage2,
                    lr_list_stage3,
                    T: int=5,
                    batch_size: int=100,
                    num_classes=2,
                    k_fold=5                    
                ) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = device
        self.T = T
        self.num_classes = num_classes
        self.lr_list_stage2 = lr_list_stage2
        self.lr_list_stage3 = lr_list_stage3
        if model_name == 'esfpnet':
            self.lung_segmenter = ESFPNetStructure('B0', 160)
            self.optimizer_1 = torch.optim.AdamW(self.lung_segmenter.parameters(), lr=1e-4)

            self.infection_segmenter2 = ESFPNetStructure('B0', 160)
            self.optimizer_3 = torch.optim.AdamW(self.infection_segmenter2.parameters(), lr=1e-4)
        elif model_name == 'pspnet':
            self.model, self.optimizer = psp_model_optimizer(layers=50, num_classes=2)
        elif model_name == 'unet':
            pretrained_unet = torch.load('./saved_model/unet_carvana_scale0.5_epoch2.pth')
            self.lung_segmenter = UNet(n_channels=1, n_classes=2)
            self.optimizer_1 = torch.optim.AdamW(self.lung_segmenter.parameters(), lr=1e-4)

            self.infection_segmenter1 = []
            self.optimizer_2 = []
            for t in range(T):
                self.infection_segmenter1.append(UNet(n_channels=1, n_classes=2))
                self.optimizer_2.append(torch.optim.AdamW(self.infection_segmenter1[t].parameters(), lr=1e-4))
            
            self.infection_segmenter2 = UNet(n_channels=3, n_classes=num_classes)
            self.optimizer_3 = torch.optim.AdamW(self.infection_segmenter2.parameters(), lr=1e-4)

            # self.infection_segmenter1 = []
            # self.optimizer_2 = []
            # for i, lr in enumerate(lr_list_stage2): # Allow all learning rates in all T optimizers
            #         self.optimizer_2.append([])
            # for t in range(T):
            #     self.infection_segmenter1.append(UNet(n_channels=1, n_classes=2))
            #     for i, lr in enumerate(lr_list_stage2): # Allow all learning rates in all T optimizers
            #         self.optimizer_2[i].append(torch.optim.AdamW(self.infection_segmenter1[t].parameters(), lr=lr))
            
            # self.infection_segmenter2 = UNet(n_channels=3, n_classes=num_classes)
            # # self.infection_segmenter2.load_state_dict(pretrained_unet)
            # self.optimizer_3 = []
            # for lr in lr_list_stage3:
            #     self.optimizer_3.append(torch.optim.AdamW(self.infection_segmenter2.parameters(), lr=lr))
                
        self.best_f1 = 0
        self.best_IOU = 0
        self.best_lr2 = 0
        self.best_lr3 = 0
        dataloader_args = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}
        # dataloader_args = {}

        self.train_dataset = ImageLoader(
                                            split='train', im_file=train_im, lung_msk_file=train_lung_msk, inf_msk_file=train_inf_msk, transform=train_data_transforms
                                        )
        
        self.val_dataset = ImageLoader(
                                            split='val', im_file=validation_im, lung_msk_file=None, inf_msk_file=validation_inf_msk, transform=val_data_transforms
                                        )
        
        # Set up cross-validation splits
        
        self.train_loader_list, self.val_loader_list = crossvalidation(self.train_dataset, k=k_fold, batch_size=batch_size)

        self.train_loader_full = DataLoader(
                                        self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
                                        )
        self.val_loader_full = DataLoader(
                                        self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
                                    )
        
        self.num_train_images = len(self.train_dataset)
        self.num_val_images = len(self.val_dataset)

        self.train_loss_history_stage1 = []
        self.train_IOU_history_stage1 = []
        self.train_f1_history_stage1 = []

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
            checkpoint = torch.load('./saved_model/stage1.pt')
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

            if train_f1 > best_f1:
                best_f1 = train_f1
                save_model(self.lung_segmenter, self.optimizer_1, './saved_model/stage1.pt')

            val_loss, val_IOU = self.validate_stage1(save_im)

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Train IOU: {train_IOU:.4f}"
                + f" Train F1 Score: {train_f1:.4f}"
            )

    def run_training_loop_stage2_3(self, num_epochs: int, load_from_disk: bool) -> None:
        if load_from_disk:
            for t in range(self.T):
                # Stage 2
                checkpoint = torch.load(f'./saved_model/stage2_{t+1}_final.pt')
                self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_to(self.optimizer_2[t], device)
            # Stage 3
            checkpoint = torch.load('./saved_model/stage3_final.pt')
            self.infection_segmenter2.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
            optimizer_to(self.optimizer_3, device)
        else: # load the best cross-validation parameters
            for t in range(self.T):
                # Stage 2
                checkpoint = torch.load(f'./saved_model/stage2_{t+1}.pt')
                self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
                optimizer_to(self.optimizer_2[t], device)
            # Stage 3
            checkpoint = torch.load('./saved_model/stage3.pt')
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
                    save_model(self.infection_segmenter1[t], self.optimizer_2[t], f'./saved_model/stage2_{t+1}_final.pt')
                save_model(self.infection_segmenter2, self.optimizer_3, './saved_model/stage3_final.pt')

    def train_stage1(self, save_im: bool) -> Tuple[float, float]:
        # Set training mode
        
        self.lung_segmenter.train()

        train_loss_meter = AverageMeter()
        train_IOU_meter = AverageMeter()
        train_f1_meter = AverageMeter()

        # loop over each minibatch

        for batch_number, (image, lung_image, lung_mask, inf_mask) in enumerate(self.train_loader_full):
            # print('Before everything to GPU', torch.cuda.memory_allocated(0))

            image = image.to(device)
            lung_mask = lung_mask.to(device)

            # print('After Data to GPU', torch.cuda.memory_allocated(0))

            n = image.shape[0]

            # Stage 1

            lung_mask = lung_mask.long()
            self.lung_segmenter = self.lung_segmenter.to(device)
            lung_logits = self.lung_segmenter(image)
            stage1_loss = self.lung_segmenter.criterion(lung_logits, lung_mask)
            self.optimizer_1.zero_grad()
            stage1_loss.backward()
            self.optimizer_1.step()
            if self.model_name == 'esfpnet':
                y_hat = torch.sigmoid(lung_logits)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'unet':
                y_hat = torch.argmax(lung_logits, dim=1)
            f1_score = BinaryF1(y_hat, lung_mask)

            # Clear from GPU

            self.lung_segmenter = self.lung_segmenter.cpu()
            image = image.detach().cpu()
            lung_logits = lung_logits.detach().cpu()
            lung_mask = lung_mask.detach().cpu()
            stage1_loss = stage1_loss.detach().cpu()
            y_hat = y_hat.detach().cpu()
            f1_score = f1_score.detach().cpu()

            # print('After stage 1 to CPU', torch.cuda.memory_allocated(0))

            train_loss_meter.update(val=float(stage1_loss.item()), n=n)

            iou = IOU(y_hat, lung_mask) # Calculate IOUs
            train_IOU_meter.update(val=float(iou), n=n)
            train_f1_meter.update(val=float(f1_score), n=n)

            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            stage1_loss = stage1_loss.detach().cpu()

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

        # loop over each minibatch

        for batch_number, (image, inf_mask) in enumerate(self.val_loader_full):
            image = image.to(device)
            n = image.shape[0]

            # Stage 1
            
            self.lung_segmenter = self.lung_segmenter.to(device)
            lung_logits = self.lung_segmenter(image)
            if self.model_name == 'esfpnet':
                lung_mask = torch.sigmoid(lung_logits)
                lung_mask = (lung_mask > 0.5) * 1
            elif self.model_name == 'unet':
                lung_mask = torch.argmax(lung_logits, dim=1)
            
            # Create the lung regions for validation set

            lung_image = torch.zeros(image.shape).double().to(device)
            for image_index in range(n):
                lung_index = lung_mask[image_index] != 0
                background_index = lung_mask[image_index] == 0
                lung_image[image_index, 0][lung_index] = image[image_index, 0][lung_index]
                lung_image[image_index, 0][background_index] = torch.min(image[image_index, 0])

            stage1_loss = 0 # No lung masks available
            
            # Clear from GPU

            self.lung_segmenter = self.lung_segmenter.cpu()
            image = image.detach().cpu()
            lung_logits = lung_logits.detach().cpu()
            lung_mask = lung_mask.detach().cpu()
            lung_image = lung_image.detach().cpu()

            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            
            # Empty GPU memory

            torch.cuda.empty_cache()

            if save_im:
                if batch_number == 0:
                    self.original_images_val = image
                    self.lung_images_val = lung_image
                    self.lung_mask_val = lung_mask
                    self.original_masks_val = inf_mask
                else:
                    self.original_images_val = np.concatenate((self.original_images_val, image))
                    self.lung_images_val = np.concatenate((self.lung_images_val, lung_image))
                    self.lung_mask_val = np.concatenate((self.lung_mask_val, lung_mask))
                    self.original_masks_val = np.concatenate((self.original_masks_val, inf_mask))

        return 0, 0 # No ground truth available for validation at this stage

    def crossval_epoch(self, num_epochs: int, load_from_disk: bool):
        best_f1 = 0
        best_IOU = 0
        for lr2 in self.lr_list_stage2:
            for lr3 in self.lr_list_stage3:
                for fold_number, (train_loader, val_loader) in enumerate(zip(self.train_loader_list, self.val_loader_list)): 
                    # Choose whether to load model and optimizer from disk
                    if load_from_disk:
                        for t in range(self.T):
                            # Stage 2
                            checkpoint = torch.load(f'./saved_model/stage2_{t+1}_final.pt')
                            self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                            self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
                            optimizer_to(self.optimizer_2[t], device)
                        # Stage 3
                        checkpoint = torch.load('./saved_model/stage3_final.pt')
                        self.infection_segmenter2.load_state_dict(checkpoint['model_state_dict'])
                        self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
                        optimizer_to(self.optimizer_3, device)
                    else:
                        self.infection_segmenter1 = []
                        self.optimizer_2 = []
                        for t in range(self.T):
                            self.infection_segmenter1.append(UNet(n_channels=1, n_classes=2))
                            self.optimizer_2.append(torch.optim.AdamW(self.infection_segmenter1[t].parameters(), lr=lr2))
                        
                        self.infection_segmenter2 = UNet(n_channels=3, n_classes=self.num_classes)
                        self.optimizer_3 = torch.optim.AdamW(self.infection_segmenter2.parameters(), lr=lr3)
                    # Run over the epochs
                    for epoch in range(num_epochs):
                        train_loss, train_IOU, train_f1 = TrainCross(train_loader, self.model_name, self.infection_segmenter1, self.infection_segmenter2,
                                                                        self.optimizer_2, self.optimizer_3, self.T, self.num_classes)
                        self.train_loss_history_cross[fold_number].append(train_loss)
                        self.train_IOU_history_cross[fold_number].append(train_IOU)
                        self.train_f1_history_cross[fold_number].append(train_f1)

                        val_loss, val_IOU, val_f1 = ValidateCross(val_loader, self.model_name, self.infection_segmenter1, self.infection_segmenter2,
                                                                    self.T, self.num_classes)
                        self.validation_loss_history_cross[fold_number].append(val_loss)
                        self.validation_IOU_history_cross[fold_number].append(val_IOU)
                        self.validation_f1_history_cross[fold_number].append(val_f1)

                        # Save best model

                        if val_f1 > best_f1:
                            best_f1 = val_f1
                            best_IOU = val_IOU
                            self.best_lr2 = lr2
                            self.best_lr3 = lr3
                            for t in range(self.T):
                                save_model(self.infection_segmenter1[t], self.optimizer_2[t], f'./saved_model/stage2_{t+1}.pt')
                            save_model(self.infection_segmenter2, self.optimizer_3, './saved_model/stage3.pt')
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
            # print('Before everything to GPU', torch.cuda.memory_allocated(0))

            lung_image = lung_image.to(device)
            inf_mask = inf_mask.to(device)

            # print('After Data to GPU', torch.cuda.memory_allocated(0))

            n = image.shape[0]
            if self.model_name == 'esfpnet':
                logits = self.model(x)
                batch_loss = ange_structure_loss(logits, masks)
                y_hat = torch.sigmoid(logits)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'pspnet':
                logits, y_hat, aux_loss, main_loss = self.model(x, masks)
                aux_weight = 0.4
                batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
            elif self.model_name == 'unet':
                lung_mask = lung_mask.long()
                inf_mask = inf_mask.long()

                # Stage 2
                
                T_probabilities = torch.zeros(self.T, n, self.num_classes, image.shape[2], image.shape[3])
                # T_preds = torch.zeros(n, self.T, image.shape[2], image.shape[3])
                stage2_loss_total = 0
                for t in range(self.T):
                    self.infection_segmenter1[t] = self.infection_segmenter1[t].to(device)
                    inf_logits1 = self.infection_segmenter1[t](lung_image)
                    prob = nn.Softmax(dim=1)(inf_logits1)
                    stage2_loss = self.infection_segmenter1[t].criterion(inf_logits1, inf_mask)
                    y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)

                    self.optimizer_2[t].zero_grad()
                    stage2_loss.backward()
                    self.optimizer_2[t].step()

                    # Clear from GPU

                    prob = prob.detach().cpu()
                    self.infection_segmenter1[t] = self.infection_segmenter1[t].cpu()
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

                self.infection_segmenter2 = self.infection_segmenter2.to(device)
                # print('After stage 3 to GPU', torch.cuda.memory_allocated(0))
                # inf_logits2 = self.infection_segmenter2(torch.concat((T_preds, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
                # inf_logits2 = self.infection_segmenter2(torch.concat((T_probabilities, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
                inf_logits2 = self.infection_segmenter2(torch.concat((lung_image, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
                stage3_loss = self.infection_segmenter2.criterion(inf_logits2, inf_mask)
                y_hat = torch.argmax(inf_logits2, dim=1)
                f1_score = BinaryF1(y_hat, inf_mask)

                self.optimizer_3.zero_grad()
                stage3_loss.backward()
                self.optimizer_3.step()

                # Clear from GPU

                self.infection_segmenter2 = self.infection_segmenter2.cpu()
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
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

            # print('After clearing', torch.cuda.memory_allocated(0))

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
        for batch_number, (image, inf_mask) in enumerate(self.val_loader_full):
            image = image.to(device)
            inf_mask = inf_mask.to(device)
            n = image.shape[0]
            if batch_number != len(self.val_loader_full) - 1:
                lung_image = torch.tensor(self.lung_images_val[index_count:index_count+n])
                index_count += n
            else:
                lung_image = torch.tensor(self.lung_images_val[index_count:])
            lung_image = lung_image.to(device)

            # inf_mask = torch.squeeze(inf_mask, dim=1) # Compensating for the transforms
            
            if self.model_name == 'esfpnet':
                logits = self.model(x)
                batch_loss = ange_structure_loss(logits, masks)
                y_hat = torch.sigmoid(logits)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'pspnet':
                logits, y_hat, aux_loss, main_loss = self.model(x, masks)
                aux_weight = 0.4
                batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
            elif self.model_name == 'unet':
                inf_mask = inf_mask.long()

                # Stage 2
                
                T_probabilities = torch.zeros(self.T, n, self.num_classes, image.shape[2], image.shape[3])
                # T_preds = torch.zeros(n, self.T, image.shape[2], image.shape[3])
                stage2_loss_total = 0
                for t in range(self.T):
                    self.infection_segmenter1[t] = self.infection_segmenter1[t].to(device)
                    # print(f'Before {t}', torch.cuda.memory_allocated(0))
                    inf_logits1 = self.infection_segmenter1[t](lung_image)
                    # print(f'After {t}', torch.cuda.memory_allocated(0))
                    prob = nn.Softmax(dim=1)(inf_logits1)
                    stage2_loss = self.infection_segmenter1[t].criterion(inf_logits1, inf_mask)
                    y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)
                    
                    # Clear from GPU

                    prob = prob.detach().cpu()
                    self.infection_segmenter1[t] = self.infection_segmenter1[t].cpu()
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

                self.infection_segmenter2 = self.infection_segmenter2.to(device)
                # inf_logits2 = self.infection_segmenter2(torch.concat((T_preds, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
                # inf_logits2 = self.infection_segmenter2(torch.concat((T_probabilities, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
                inf_logits2 = self.infection_segmenter2(torch.concat((lung_image, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
                stage3_loss = self.infection_segmenter2.criterion(inf_logits2, inf_mask)
                y_hat = torch.argmax(inf_logits2, dim=1)
                f1_score = BinaryF1(y_hat, inf_mask)

                # Clear from GPU

                self.infection_segmenter2 = self.infection_segmenter2.cpu()
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
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

            if save_im:
                if batch_number == 0:
                    self.var = sam_var
                    self.pred_ent = pred_ent
                    self.predictions_val = y_hat
                    if self.model_name == 'unet':
                        self.probability_val = prob
                else:
                    self.var = np.concatenate((self.var, sam_var))
                    self.pred_ent = np.concatenate((self.pred_ent, pred_ent))
                    self.predictions_val = np.concatenate((self.predictions_val, y_hat))
                    if self.model_name == 'unet':
                        self.probability_val = np.concatenate((self.probability_val, prob))

        return val_loss_meter.avg, val_IOU_meter.avg, val_f1_meter.avg
    

    def Predict(self, image, load_model_from_disk: bool):
        assert image.ndim == 4
        if load_model_from_disk:
            # Stage 1
            checkpoint = torch.load('./saved_model/stage1.pt')
            self.lung_segmenter.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_1.load_state_dict(checkpoint['optimizer_state_dict'])
            for t in range(self.T):
                # Stage 2
                checkpoint = torch.load(f'./saved_model/stage2_{t+1}_final.pt')
                self.infection_segmenter1[t].load_state_dict(checkpoint['model_state_dict'])
                self.optimizer_2[t].load_state_dict(checkpoint['optimizer_state_dict'])
            # Stage 3
            checkpoint = torch.load('./saved_model/stage3_final.pt')
            self.infection_segmenter2.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lung_segmenter = self.lung_segmenter.to(device)
        lung_logits = self.lung_segmenter(image)
        if self.model_name == 'esfpnet':
            lung_mask = torch.sigmoid(lung_logits)
            lung_mask = (lung_mask > 0.5) * 1
        elif self.model_name == 'unet':
            lung_mask = torch.argmax(lung_logits, dim=1)
        
        # Create the lung regions for validation set

        lung_image = torch.zeros(image.shape).double().to(device)
        lung_index = lung_mask[0] != 0
        background_index = lung_mask[0] == 0
        lung_image[0, 0][lung_index] = image[0, 0][lung_index]
        lung_image[0, 0][background_index] = torch.min(image[0, 0])

        T_probabilities = torch.zeros(self.T, 1, self.num_classes, image.shape[2], image.shape[3])
        # T_preds = torch.zeros(n, self.T, image.shape[2], image.shape[3])
        for t in range(self.T):
            self.infection_segmenter1[t] = self.infection_segmenter1[t].to(device)
            # print(f'Before {t}', torch.cuda.memory_allocated(0))
            inf_logits1 = self.infection_segmenter1[t](lung_image)
            # print(f'After {t}', torch.cuda.memory_allocated(0))
            prob = nn.Softmax(dim=1)(inf_logits1)
            y_hat_inf_1 = torch.argmax(inf_logits1, dim=1)
            
            # Clear from GPU

            prob = prob.detach().cpu()
            self.infection_segmenter1[t] = self.infection_segmenter1[t].cpu()
            inf_logits1 = inf_logits1.detach().cpu()
            y_hat_inf_1 = y_hat_inf_1.detach().cpu()
            T_probabilities[t] = prob # Store T probabilities for each image in the batch
            # T_preds[:, t] = y_hat_inf_1

        # T_preds = T_preds.detach()
        T_probabilities = T_probabilities.detach()
        sam_var = sample_variance(T_probabilities)
        pred_ent = predictive_entropy(T_probabilities)
        sam_var = sam_var.to(device)
        pred_ent = pred_ent.to(device)

        # Stage 3

        self.infection_segmenter2 = self.infection_segmenter2.to(device)
        # inf_logits2 = self.infection_segmenter2(torch.concat((T_preds, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
        # inf_logits2 = self.infection_segmenter2(torch.concat((T_probabilities, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
        inf_logits2 = self.infection_segmenter2(torch.concat((lung_image, torch.unsqueeze(sam_var, dim=1), torch.unsqueeze(pred_ent, dim=1)), dim=1))
        y_hat = torch.argmax(inf_logits2, dim=1)

        # Clear from GPU

        self.infection_segmenter2 = self.infection_segmenter2.cpu()
        inf_logits2 = inf_logits2.detach().cpu()
        stage3_loss = stage3_loss.detach().cpu()
        f1_score = f1_score.detach().cpu()
        lung_image = lung_image.detach().cpu()

        return y_hat
