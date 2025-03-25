import os
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from src.training.train_utils import *
from src.data.image_loader import *
from src.training.metrics import *
from src.models.pspnet.pspnet import *
from torch.utils.data import DataLoader
from typing import List, Tuple
from src.training.model_optimizer import get_model_optimizer
import gc


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer:
    """Class that stores model training metadata."""

    def __init__(
                    self,
                    model_name: str,
                    train_data_transforms_common: transforms.Compose,
                    train_data_transforms_image: transforms.Compose,
                    val_data_transforms: transforms.Compose,
                    batch_size: int=100,
                    num_classes=2,
                    B='B0'
                ) -> None:
        self.device = device
        self.model_dir = os.path.join('./saved_model', model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.num_classes = num_classes
        
        self.lung_segmenter, self.infection_segmenter, self.optimizer1, self.optimizer2 = \
            get_model_optimizer(model_name, B)

        self.model_name = model_name
        dataloader_args = {"num_workers": 4, "pin_memory": True} if torch.cuda.is_available() else {}

        self.train_dataset = ImageLoader(data_dir="dataset",
                                         split='train',
                                         transform_common=train_data_transforms_common,
                                         transform_image=train_data_transforms_image
                                        )
        
        self.val_dataset = ImageLoader(data_dir="dataset",
                                         split='validation',
                                         transform_image=val_data_transforms
                                        )

        # Drop last batch if last batch size is 1 to keep batchnorm from breaking.
        self.num_train_images = len(self.train_dataset)
        self.num_val_images = len(self.val_dataset)
        drop_last_train = self.num_train_images % batch_size == 1
        drop_last_val = self.num_val_images % batch_size == 1
        
        self.train_loader = DataLoader(
                                        self.train_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, **dataloader_args,
                                        drop_last=drop_last_train
                                        )
        self.val_loader = DataLoader(
                                        self.val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False, **dataloader_args,
                                        drop_last=drop_last_val
                                    )

        self.train_loss_history = []
        self.validation_loss_history = []
        self.train_IOU_history = []
        self.validation_IOU_history = []
        self.train_f1_history = []
        self.validation_f1_history = []
        self.best_f1 = 0
        self.saved_model_dir = os.path.join("saved_model", model_name)
        os.makedirs(self.saved_model_dir, exist_ok=True)

        self.lung_segmenter = self.lung_segmenter.to(device)
        self.infection_segmenter = self.infection_segmenter.to(device)

    def run_training_loop(self, num_epochs: int) -> None:
        """Train for num_epochs, and validate after every epoch."""
        for epoch_idx in range(num_epochs):
            # Train
            train_loss, train_IOU, train_f1, train_accuracy = \
                train(self.train_loader, self.lung_segmenter,
                      self.infection_segmenter, self.optimizer1,
                      self.optimizer2, self.model_name, device)
            self.train_loss_history.append(train_loss)
            self.train_IOU_history.append(train_IOU)
            self.train_f1_history.append(train_f1)

            # Validate
            val_loss, val_IOU, val_f1, val_accuracy = \
                validate(self.val_loader, self.lung_segmenter,
                      self.infection_segmenter, self.model_name, device)
            self.validation_loss_history.append(val_loss)
            self.validation_IOU_history.append(val_IOU)
            self.validation_f1_history.append(val_f1)

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                save_model(self.lung_segmenter, self.optimizer1,
                           f"{self.saved_model_dir}/lung_segmenter.pt")
                save_model(self.infection_segmenter, self.optimizer2,
                           f"{self.saved_model_dir}/infection_segmenter.pt")
                
            print(f"Epoch {epoch_idx + 1}:")
            print(f"\tTrain Loss: {train_loss:.4f}")
            print(f"\tTrain F1-Score: {train_f1:.4f}")
            print(f"\tTrain Accuracy: {train_accuracy:.4f}")
            print(f"\tValidation Loss: {val_loss:.4f}")
            print(f"\tValidation F1-Score: {val_f1:.4f}")
            print(f"\tValidation Accuracy: {val_accuracy:.4f}")

        # Empty GPU after all epochs
        self.lung_segmenter = self.lung_segmenter.cpu()
        self.infection_segmenter = self.infection_segmenter.cpu()
        gc.collect()
        torch.cuda.empty_cache()
