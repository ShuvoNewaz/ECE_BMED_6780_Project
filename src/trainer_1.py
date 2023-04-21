import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.image_loader import *
from src.metrics import *
from src.avg_meter import AverageMeter, SegmentationAverageMeter
from src.esfpnet import ESFPNetStructure
from src.pspnet import *
from src.unet_model import *
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
        elif model_name == 'unet':
            if seg_type == 'lung':
                self.model = UNet(1, 2)
            else:
                self.model = UNet(1, 2)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.model_name = model_name
        self.model = self.model.to(device)
        self.seg_type = seg_type
        dataloader_args = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
        dataloader_args = {}

        self.train_dataset = ImageLoaderFile(
                                            im_file=train_im, msk_file=train_msk, transform=train_data_transforms
                                        )
        self.val_dataset = ImageLoaderFile(
                                            im_file=validation_im, msk_file=validation_msk, transform=val_data_transforms
                                        )

        self.train_loader = DataLoader(
                                        self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
                                        )
        self.val_loader = DataLoader(
                                        self.val_dataset, batch_size=batch_size, shuffle=False, **dataloader_args
                                    )

        # self.optimizer = optimizer
        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_IOU_history = []
        self.validation_IOU_history = []

        # load the model from the disk if it exists
        if os.path.exists(model_dir) and load_from_disk:
            checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.model.train()

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
            save_im = epoch_idx == num_epochs-1 # Save images only from the last epoch
            train_loss, train_IOU = self.train_epoch(save_im)

            self.train_loss_history.append(train_loss)
            self.train_IOU_history.append(train_IOU)

            val_loss, val_IOU = self.validate(save_im)
            self.validation_loss_history.append(val_loss)
            self.validation_IOU_history.append(val_IOU)

            print(
                f"Epoch:{epoch_idx + 1}"
                + f" Train Loss:{train_loss:.4f}"
                + f" Val Loss: {val_loss:.4f}"
                + f" Train IOU: {train_IOU:.4f}"
                + f" Validation IOU: {val_IOU:.4f}"
            )

    def train_epoch(self, save_im: bool=False) -> Tuple[float, float]:
        """Implements the main training loop."""
        self.model.train()

        train_loss_meter = AverageMeter()
        train_IOU_meter = AverageMeter()

        # loop over each minibatch

        for batch_number, (x, masks) in enumerate(self.train_loader):
            # print('Before x-mask to GPU', torch.cuda.memory_allocated(0))
            # print(x.shape, masks.shape)
            x = x.to(device)
            masks = masks.to(device)
            masks[masks != 0] = 1 # Reduce to a binary classification problem

            n = x.shape[0]
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
                masks = masks.long()
                logits = self.model(x)
                prob = nn.Softmax(dim=1)(logits)
                batch_loss = self.model.criterion(logits, masks)
                y_hat = torch.argmax(logits, dim=1)

            train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)
            
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            # Empty GPU memory

            x = x.detach().cpu()
            masks = masks.detach().cpu()
            logits = logits.detach().cpu()
            y_hat = y_hat.detach().cpu()
            iou = IOU(y_hat, masks) # Calculate IOUs
            train_IOU_meter.update(val=float(iou), n=n)
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

            if save_im:
                if batch_number == 0:
                    self.original_images_train = x
                    self.predictions_train = y_hat
                    if self.model_name == 'unet':
                        prob = prob.detach().cpu()
                        self.probability_train = prob
                    if self.seg_type != 'lung':
                        self.original_masks_train = masks
                else:
                    self.original_images_train = torch.concat((self.original_images_train, x))
                    self.predictions_train = torch.concat((self.predictions_train, y_hat))
                    if self.model_name == 'unet':
                        prob = prob.detach().cpu()
                        self.probability_train = np.concatenate((self.probability_train, prob))
                    if self.seg_type != 'lung':
                        self.original_masks_train = torch.concat((self.original_masks_train, masks))

        return train_loss_meter.avg, train_IOU_meter.avg

    def validate(self, save_im: bool=False) -> Tuple[float, float]:
        """Evaluate on held-out split (either val or test)"""
        self.model.eval()

        val_loss_meter = AverageMeter()
        val_IOU_meter = AverageMeter()

        # loop over whole val set

        for batch_number, (x, masks) in enumerate(self.val_loader):
            x = x.to(device)
            masks = masks.to(device)
            masks[masks != 0] = 1 # Reduce to a binary classification problem

            n = x.shape[0]
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
                masks = masks.long()
                logits = self.model(x)
                prob = nn.Softmax(dim=1)(logits)
                batch_loss = self.model.criterion(logits, masks)
                y_hat = torch.argmax(logits, dim=1)
            
            val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

            # Empty GPU memory

            x = x.detach().cpu()
            masks = masks.detach().cpu()
            logits = logits.detach().cpu()
            y_hat = y_hat.detach().cpu()
            iou = IOU(y_hat, masks) # Calculate IOUS
            val_IOU_meter.update(val=float(iou), n=n)
            if self.model_name == 'pspnet':
                main_loss = main_loss.detach().cpu()
                aux_loss = aux_loss.detach().cpu()
            
            batch_loss = batch_loss.detach().cpu()
            torch.cuda.empty_cache()

            # Store images for viewing

            if save_im:
                if batch_number == 0:
                    self.original_images_val = x
                    self.predictions_val = y_hat
                    if self.model_name == 'unet':
                        prob = prob.detach().cpu()
                        self.probability_val = prob
                    if self.seg_type != 'lung':
                        self.original_masks_val = masks
                else:
                    self.original_images_val = torch.concat((self.original_images_val, x))
                    self.predictions_val = torch.concat((self.predictions_val, y_hat))
                    if self.model_name == 'unet':
                        prob = prob.detach().cpu()
                        self.probability_val = np.concatenate((self.probability_val, prob))
                    if self.seg_type != 'lung':
                        self.original_masks_val = torch.concat((self.original_masks_val, masks))

        return val_loss_meter.avg, val_IOU_meter.avg

    def plot_loss_history(self) -> None:
        """Plots the loss history"""
        epoch_idxs = range(len(self.train_loss_history))
        plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
        plt.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
        plt.title("Loss history")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epochs")

    def plot_IOU_history(self) -> None:
        """Plots the IOU history"""
        epoch_idxs = range(len(self.train_IOU_history))
        plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
        plt.plot(epoch_idxs, self.train_IOU_history, "-b", label="training")
        plt.plot(epoch_idxs, self.validation_IOU_history, "-r", label="validation")
        plt.title("IOU history")
        plt.legend()
        plt.ylabel("IOU")
        plt.xlabel("Epochs")
        

# class MultiLabelTrainer:
#     """Class that stores model training metadata."""

#     def __init__(
#         self,
#         data_dir: str,
#         model: MultilabelResNet18,
#         optimizer: Optimizer,
#         model_dir: str,
#         train_data_transforms: transforms.Compose,
#         val_data_transforms: transforms.Compose,
#         batch_size: int = 100,
#         load_from_disk: bool = True,
#         cuda: bool = False,
#     ) -> None:
#         self.model_dir = model_dir

#         self.model = model

#         self.cuda = cuda
#         if cuda:
#             self.model.cuda()

#         dataloader_args = {"num_workers": 1, "pin_memory": True} if cuda else {}

#         self.root_dir = os.path.split(data_dir)[0] if os.path.split(data_dir)[1] != '' else os.path.split(os.path.split(data_dir)[0])[0]
#         self.train_csv = os.path.join(self.root_dir, 'scene_attributes_train.csv')
#         self.test_csv = os.path.join(self.root_dir, 'scene_attributes_test.csv')

#         self.train_dataset = MultiLabelImageLoader(
#             data_dir, labels_csv=self.train_csv, split="train", transform=train_data_transforms
#         )
#         self.train_loader = DataLoader(
#             self.train_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
#         )

#         self.val_dataset = MultiLabelImageLoader(
#             data_dir, labels_csv=self.test_csv, split="test", transform=val_data_transforms
#         )
#         self.val_loader = DataLoader(
#             self.val_dataset, batch_size=batch_size, shuffle=True, **dataloader_args
#         )

#         self.optimizer = optimizer

#         self.train_loss_history = []
#         self.validation_loss_history = []
#         self.train_accuracy_history = []
#         self.validation_accuracy_history = []

#         # load the model from the disk if it exists
#         if os.path.exists(model_dir) and load_from_disk:
#             checkpoint = torch.load(os.path.join(self.model_dir, "checkpoint.pt"))
#             self.model.load_state_dict(checkpoint["model_state_dict"])
#             self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#         self.model.train()

#     def save_model(self) -> None:
#         """
#         Saves the model state and optimizer state on the dict
#         """
#         torch.save(
#             {
#                 "model_state_dict": self.model.state_dict(),
#                 "optimizer_state_dict": self.optimizer.state_dict(),
#             },
#             os.path.join(self.model_dir, "checkpoint.pt"),
#         )

#     def run_training_loop(self, num_epochs: int) -> None:
#         """Train for num_epochs, and validate after every epoch."""
#         # best_accuracy = 0
#         for epoch_idx in range(num_epochs):

#             train_loss, train_acc = self.train_epoch()

#             self.train_loss_history.append(train_loss)
#             self.train_accuracy_history.append(train_acc)

#             val_loss, val_acc = self.validate()
#             self.validation_loss_history.append(val_loss)
#             self.validation_accuracy_history.append(val_acc)
#             # if val_acc > best_accuracy:
#             #     best_accuracy = val_acc
#             #     save_trained_model_weights(self.model, out_dir="./src/vision")

#             print(
#                 f"Epoch:{epoch_idx + 1}"
#                 + f" Train Loss:{train_loss:.4f}"
#                 + f" Val Loss: {val_loss:.4f}"
#                 + f" Train Accuracy: {train_acc:.4f}"
#                 + f" Validation Accuracy: {val_acc:.4f}"
#             )

#     def train_epoch(self) -> Tuple[float, float]:
#         """Implements the main training loop."""
#         self.model.train()

#         train_loss_meter = AverageMeter("train loss")
#         train_acc_meter = AverageMeter("train accuracy")

#         # loop over each minibatch
#         for (x, y) in self.train_loader:
#             if self.cuda:
#                 x = x.cuda()
#                 y = y.cuda()

#             n = x.shape[0]
#             logits = self.model(x)
#             batch_acc = compute_multilabel_accuracy(logits, y)
#             train_acc_meter.update(val=batch_acc, n=n)

#             batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
#             train_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

#             self.optimizer.zero_grad()
#             batch_loss.backward()
#             self.optimizer.step()

#         return train_loss_meter.avg, train_acc_meter.avg

#     def validate(self) -> Tuple[float, float]:
#         """Evaluate on held-out split (either val or test)"""
#         self.model.eval()

#         val_loss_meter = AverageMeter("val loss")
#         val_acc_meter = AverageMeter("val accuracy")

#         # loop over whole val set
#         for (x, y) in self.val_loader:
#             if self.cuda:
#                 x = x.cuda()
#                 y = y.cuda()

#             n = x.shape[0]
#             logits = self.model(x)

#             batch_acc = compute_multilabel_accuracy(logits, y)
#             val_acc_meter.update(val=batch_acc, n=n)

#             batch_loss = compute_loss(self.model, logits, y, is_normalize=True)
#             val_loss_meter.update(val=float(batch_loss.cpu().item()), n=n)

#         return val_loss_meter.avg, val_acc_meter.avg

#     def plot_loss_history(self) -> None:
#         """Plots the loss history"""
#         plt.figure()
#         epoch_idxs = range(len(self.train_loss_history))
#         plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
#         plt.plot(epoch_idxs, self.train_loss_history, "-b", label="training")
#         plt.plot(epoch_idxs, self.validation_loss_history, "-r", label="validation")
#         plt.title("Loss history")
#         plt.legend()
#         plt.ylabel("Loss")
#         plt.xlabel("Epochs")
#         plt.show()

#     def plot_accuracy(self) -> None:
#         """Plots the accuracy history"""
#         plt.figure()
#         epoch_idxs = range(len(self.train_accuracy_history))
#         plt.xticks(epoch_idxs[::5], epoch_idxs[::5])
#         plt.plot(epoch_idxs, self.train_accuracy_history, "-b", label="training")
#         plt.plot(epoch_idxs, self.validation_accuracy_history, "-r", label="validation")
#         plt.title("Accuracy history")
#         plt.legend()
#         plt.ylabel("Accuracy")
#         plt.xlabel("Epochs")
#         plt.show()