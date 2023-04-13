import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from data.image_loader import ImageLoader
from utils.metrics import ange_structure_loss, BinaryF1
from utils.avg_meter import AverageMeter, SegmentationAverageMeter
from models import ESFPNetStructure, PSPNet, UNet, EnsembleNet
from transforms.data_transforms import get_train_transforms, get_val_transforms

from typing import List, Tuple
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

import numpy as np
import json

import argparse
from mmcv.utils import Config, DictAction

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
from utils.logger import get_logger

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
                    test_im: str,
                    test_msk: str = '',
                    batch_size: int=100,
                    load_from_disk: bool = True,
                    model_args: dict = dict(),
                    logger = None,
                    predict=False
                ) -> None:
        self.logger = logger or get_logger()
        self.device = device
        self.model_dir = model_dir
        if model_name == 'esfpnet':
            self.model = ESFPNetStructure(**model_args)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        elif model_name == 'pspnet':
            self.model, self.optimizer = psp_model_optimizer(layers=50, num_classes=2)
        elif model_name == 'unet':
            self.model = UNet(**model_args)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        elif model_name == 'ensemble':
            self.model = EnsembleNet(**model_args)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        else: raise NotImplementedError
        self.model_name = model_name
        self.model = self.model.to(device)
        self.seg_type = seg_type
        dataloader_args = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}

        if predict:
            self.test_dataset = ImageLoader(
                im_file=test_im, msk_file=test_msk, transform=val_data_transforms)
            self.test_loader = DataLoader(
                                            self.test_dataset, batch_size=1, shuffle=False, **dataloader_args
                                        )
        else:
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
        self.train_f1_history = []
        self.validation_f1_history = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk and not predict:
            # checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            # self.model.load_state_dict(checkpoint["model_state_dict"])
            # self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint = torch.load(os.path.join(self.model_dir, "model.pt"))
            self.model.load_state_dict(checkpoint)
            optimizer_state_dict = torch.load(os.path.join(self.model_dir, "optimizer.pt"))
            self.optimizer.load_state_dict(optimizer_state_dict)
        elif predict:
            model_file = os.path.join(self.model_dir, "best.pt")
            if os.path.isfile(model_file):
                checkpoint = torch.load(os.path.join(self.model_dir, "best.pt"))
                self.model.load_state_dict(checkpoint)
            else:
                self.logger.warning(f"{model_file} not found")
        self.model.train()
        self.original_images = [] # Store the original validation images
        self.original_masks = [] # Store the ground truth of validation set

        self.best_loss = np.inf
        self.best_state_dict = self.model.state_dict()

    def save_model(self, directory) -> None:
        """
        Saves the model state and optimizer state on the dict
        """
        # torch.save(
        #     {
        #         "model_state_dict": self.model.state_dict(),
        #         "optimizer_state_dict": self.optimizer.state_dict(),
        #         "best_state_dict": self.best_state_dict,
        #     },
        #     dir,
        # )
        torch.save(self.model.state_dict(), os.path.join(directory, "model.pt"))
        torch.save(self.best_state_dict, os.path.join(directory, "best.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(directory, "optimizer.pt"))

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):
            train_loss, train_f1 = self.train_epoch()

            self.train_loss_history.append(train_loss)
            self.train_f1_history.append(train_f1)

            val_loss, val_f1 = self.validate()
            self.validation_loss_history.append(val_loss)
            self.validation_f1_history.append(val_f1)

            self.logger.info(
                f"Epoch:{epoch_idx + 1}/{num_epochs}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train F1: {train_f1:.4f}"
                + f" Validation F1: {val_f1:.4f}"
            )
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state_dict = self.model.state_dict()


    def train_epoch(self) -> Tuple[float, float]:
        """Implements the main training loop."""
        self.model.train()

        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        nbatch = len(self.train_loader)
        # loop over each minibatch
        for i, (x, masks) in enumerate(self.train_loader, start=1):
            # print('Before x-mask to GPU', torch.cuda.memory_allocated(0))
            x = x.to(device)
            masks = masks.to(device)
            masks[masks != 0] = 1 # Reduce to a binary classification problem

            n = x.shape[0]
            

            if self.model_name == 'esfpnet':
                logits = self.model(x)
                batch_loss = ange_structure_loss(logits, masks)
            elif self.model_name == 'pspnet':
                logits, y_hat, aux_loss, main_loss = self.model(x, masks)
                aux_weight = 0.4
                batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
            elif self.model_name == 'unet':
                logits = self.model(x)
                batch_loss = self.model.criterion(logits, masks)
            else: raise NotImplementedError
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            batch_f1 = BinaryF1(logits, masks)
            train_acc_meter.update(val=float(batch_f1.cpu().item()), n=n)
            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            if i % 5 == 0:
                self.logger.info(f"Batch {i}/{nbatch} | loss={batch_loss.cpu().item()} | F1={batch_f1.cpu().item()}")

        return train_loss_meter.avg, train_acc_meter.avg

    def validate(self) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()

        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()

        # loop over whole val set
        for (x, masks) in self.val_loader:
            x = x.to(device)
            masks = masks.to(device)
            masks[masks != 0] = 1 # Reduce to a binary classification problem

            n = x.shape[0]


            if self.model_name == 'esfpnet':
                logits = self.model(x)
                batch_loss = ange_structure_loss(logits, masks)
            elif self.model_name == 'pspnet':
                logits, y_hat, aux_loss, main_loss = self.model(x, masks)
                aux_weight = 0.4
                batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
            elif self.model_name == 'unet':
                logits = self.model(x)
                batch_loss = self.model.criterion(logits, masks)
            else:
                raise NotImplementedError
            
            val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            batch_f1 = BinaryF1(logits, masks)
            val_acc_meter.update(val=float(batch_f1.cpu().item()), n=n)
            torch.cuda.empty_cache()

        return val_loss_meter.avg, val_acc_meter.avg

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()
        with torch.no_grad():
            self.model.eval()
            original_images = []
            predictions = []
            # loop over whole val set

            f1_meter = AverageMeter()
            for x, masks in self.test_loader:
                x = x.to(device)
                n = x.shape[0]

                if self.model_name in ['esfpnet', 'unet', 'ensemble']:
                    logits = self.model(x)
                    prob = torch.sigmoid(logits)
                elif self.model_name == 'pspnet':
                    # TODO: fix this part
                    logits, y_hat, aux_loss, main_loss = self.model(x, masks)
                    aux_weight = 0.4
                    batch_loss = torch.mean(main_loss) + aux_weight * torch.mean(aux_loss)
                    prob = torch.sigmoid(logits)
                else: raise NotImplementedError

                batch_f1 = BinaryF1(logits, masks)
                f1_meter.update(val=float(batch_f1.cpu().item()), n=n)

                
                prediction = (prob > 0.5).int().cpu().numpy()

                x = x.squeeze(1).detach().cpu().numpy()

                # Store images for viewing

                original_images.append(x)
                predictions.append(prediction)
        original_images = np.concatenate(original_images, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        if self.test_dataset.msk_file:
            self.logger.info(f"F1 score {f1_meter.avg}")
        else:
            self.logger.warning("Test mask not found, skip evaluation")
        return original_images, predictions

    def plot_loss_history(self, ax) -> None:
        """Plots the loss history"""
        epoch_idxs = range(len(self.train_loss_history))
        ax.set_xticks(epoch_idxs[::5], epoch_idxs[::5])
        ax.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
        ax.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
        ax.set_title("Loss history")
        ax.legend(loc="best")
        ax.set_ylabel("Loss")
        ax.set_xlabel("Epochs")

    def plot_f1(self, ax) -> None:
        """Plots the accuracy history"""
        epoch_idxs = range(len(self.train_f1_history))
        ax.set_xticks(epoch_idxs[::5], epoch_idxs[::5])
        ax.plot(epoch_idxs, self.train_f1_history, "-b", label="training")
        ax.plot(epoch_idxs, self.validation_f1_history, "-r", label="validation")
        ax.set_title("F1 history")
        ax.legend(loc="best")
        ax.set_ylabel("F1")
        ax.set_xlabel("Epochs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='config/esfpnet.py', type=str)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    os.makedirs(cfg.exp, exist_ok=True)
    if args.predict:
        logger = get_logger()
    else:
        logger = get_logger(os.path.join(cfg.exp, 'train.log'))
    logger.info(json.dumps(dict(cfg), indent=4))

    runner = Trainer(model_name=cfg.model.model_name,
                 model_dir=cfg.exp,
                 train_data_transforms=get_train_transforms(),
                 val_data_transforms=get_val_transforms(),
                 seg_type=cfg.seg_type,
                 train_im=cfg.data.train.im,
                 train_msk=cfg.data.train.msk,
                 validation_im=cfg.data.val.im,
                 validation_msk=cfg.data.val.msk,
                 test_msk=cfg.data.test.msk,
                 test_im=cfg.data.test.im,
                 batch_size=cfg.batch_size,
                 load_from_disk=cfg.load_from_disk,
                 model_args=cfg.model.args,
                 logger = logger,
                 predict=args.predict)
    if args.predict:
        original_images, predictions = runner.predict()
        N = original_images.shape[0]
        
        fig = plt.figure(figsize=(16, 4))
        for idx in range(N):
            ax = plt.subplot(2, N, idx+1)
            plt.imshow(original_images[idx], cmap='gray')
            plt.axis('off')
            ax = plt.subplot(2, N, N+idx+1)
            plt.imshow(predictions[idx], cmap='gray')
            plt.axis('off')
        fig.savefig(os.path.join(cfg.exp, "predictions.png"))

    else:
        runner.run_training_loop(num_epochs=cfg.num_epochs)
        runner.save_model(cfg.exp)
        
        fig = plt.figure(figsize=(16, 9))
        runner.plot_loss_history(plt.subplot(1, 2, 1))
        runner.plot_f1(plt.subplot(1, 2, 2))
        fig.savefig(os.path.join(cfg.exp, "history.png"))
