import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.image_loader import *
from src.metrics import *
from src.avg_meter import AverageMeter, SegmentationAverageMeter
from src.esfpnet import ESFPNetStructure
from src.pspnet import *
from typing import List, Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
                    self,
                    model_name: str,
                    model_dir: str,
                    train_data_transforms: transforms.Compose,
                    val_data_transforms: transforms.Compose,
                    seg_type: str,
                    train_im: str,
                    train_msk: str,
                    validation_im: str,
                    validation_msk: str,
                    batch_size: int=100,
                    load_from_disk: bool = True,
                ) -> None:
        self.device = device
        self.model_dir = model_dir
        if model_name == 'esfpnet':
            self.model = ESFPNetStructure('B0', 160)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        elif model_name == 'pspnet':
            self.model, self.optimizer = psp_model_optimizer(layers=50, num_classes=2)
        self.model_name = model_name
        self.model = self.model.to(device)
        self.seg_type = seg_type
        dataloader_args = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}

        self.train_dataset = ImageLoader(
                                            im_file=train_im, msk_file=train_msk, transform=train_data_transforms
                                        )
        self.val_dataset = ImageLoader(
                                            im_file=validation_im, msk_file=validation_msk, transform=val_data_transforms
                                        )

        self.train_loader = DataLoader(
                                        self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
                                        )
        self.val_loader = DataLoader(
                                        self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
                                    )

        # self.optimizer = optimizer
        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_accuracy_history = []
        self.validation_accuracy_history = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()
        self.original_images = [] # Store the original validation images
        self.original_masks = [] # Store the ground truth of validation set
        self.predictions = [] # Store the predicted images

    def save_model(self, dir) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            dir,
        )

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):
            train_loss = self.train_epoch()

            self.train_loss_history.append(train_loss)
            # self.train_accuracy_history.append(train_acc)

            val_loss = self.validate()
            self.validation_loss_history.append(val_loss)
            # self.validation_accuracy_history.append(val_acc)

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                # + f" Train Accuracy: {train_acc:.4f}"
                # + f" Validation Accuracy: {val_acc:.4f}"
            )

    def train_epoch(self) -> Tuple[float, float]:
        """Implements the main training loop."""
        self.model.train()

        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        # loop over each minibatch
        for (x, masks) in self.train_loader:
            # print('Before x-mask to GPU', torch.cuda.memory_allocated(0))
            x = x.to(device)
            masks = masks.to(device)
            masks[masks != 0] = 1 # Reduce to a binary classification problem

            n = x.shape[0]
            
            # batch_acc = IOU(logits, masks, 4, 255)
            # train_acc_meter.update(val=batch_acc, n=n)
            # print(batch_acc.shape)

            if self.model_name == 'esfpnet':
                logits = self.model(x)
                batch_loss = ange_structure_loss(logits, masks)
            elif self.model_name == 'pspnet':
                logits, y_hat, aux_loss, main_loss = self.model(x, masks)
                aux_weight = 0.4
                batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)

            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Empty GPU memory

            x = x.detach().cpu()
            masks = masks.detach().cpu()
            logits = logits.detach().cpu()
            if self.model_name == 'pspnet':
                y_hat = y_hat.detach().cpu()
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

        return train_loss_meter.avg#, train_acc_meter.avg

    def validate(self) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()

        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()

        self.original_images = []
        self.original_masks = []
        self.predictions = []
        # loop over whole val set
        for (x, masks) in self.val_loader:
            x = x.to(device)
            masks = masks.to(device)
            masks[masks != 0] = 1 # Reduce to a binary classification problem

            n = x.shape[0]

            # batch_acc = IOU(logits, masks, 4, 255)
            # val_acc_meter.update(val=batch_acc, n=n)

            if self.model_name == 'esfpnet':
                logits = self.model(x)
                batch_loss = ange_structure_loss(logits, masks)
                y_hat = torch.sigmoid(logits)
                y_hat = (y_hat > 0.5) * 1
            elif self.model_name == 'pspnet':
                logits, y_hat, aux_loss, main_loss = self.model(x, masks)
                aux_weight = 0.4
                batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
            
            val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            # Empty GPU memory

            x = x.detach().cpu()
            masks = masks.detach().cpu()
            logits = logits.detach().cpu()
            y_hat = y_hat.detach().cpu()
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

            # Store images for viewing

            self.original_images.append(x)
            if self.seg_type != 'lung':
                self.original_masks.append(masks)
            self.predictions.append(y_hat)

        return val_loss_meter.avg#, val_acc_meter.avg

    def plot_loss_history(self) -> None:
        """Plots the loss history"""
        plt.figure()
        epoch_idxs = range(len(self.train_loss_history))
        plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
        plt.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.show()

    def plot_accuracy(self) -> None:
        """Plots the accuracy history"""
        plt.figure()
        epoch_idxs = range(len(self.train_loss_history))
        plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
        plt.plot(epoch_idxs, self.train_accuracy_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_accuracy_history, "-r", label="validation")
        plt.title("Accuracy history")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epochs")
        plt.show()