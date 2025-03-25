import os
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils import data
import nibabel as nib
import numpy as np
from typing import List, Tuple
from src.data.data_utils import *


class ImageLoader(data.Dataset):
    def __init__(self, data_dir, split: str, transform_common: Compose=None, transform_image: Compose=None) -> None:
        """
        args:
            root_dir: Root working directory
            split:
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(self.data_dir, self.split)
        self.im_file = os.path.join(self.split_dir, 'im.nii.gz')
        if split != "test":
            self.lung_msk_file = os.path.join(self.split_dir, 'lung_mask.nii.gz')
            self.inf_msk_file = os.path.join(self.split_dir, 'mask.nii.gz')
        self.transform_common = transform_common
        self.transform_image = transform_image
        self.dataset = self.load_images_with_masks()

    def load_images_with_masks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the list of tuples containing the image and the mask
        of the dataset.
        """
        dataset = []
        images = nib.load(self.im_file).get_fdata()
        if self.split != "test":
            lung_masks = nib.load(self.lung_msk_file).get_fdata()
            inf_masks = nib.load(self.inf_msk_file).get_fdata()
        else:
            lung_masks = np.zeros(images.shape) # Placeholder for missing validation masks
            inf_masks = np.zeros(images.shape) # Placeholder for missing validation masks
        
        for i in range(images.shape[2]):
            dataset.append((images[:, :, i], lung_masks[:, :, i], inf_masks[:, :, i]))

        return dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        image, lung_mask, inf_mask = self.dataset[index]
        
        lung_image = extract_lung_regions(image, lung_mask, np.zeros(image.shape))

        # Facilitate transformation of masks

        images_formatted = np.concatenate((np.expand_dims(image, 2), np.expand_dims(lung_image, 2)), 2)
        image_and_mask = np.concatenate((images_formatted, np.expand_dims(lung_mask, 2), np.expand_dims(inf_mask, 2)), 2)

        # Add rotation and flips

        if self.transform_common:
            image, lung_image, lung_mask, inf_mask = self.transform_common(image_and_mask)
        
        # Add noise and jitters

        images_formatted = np.concatenate((np.expand_dims(image, 2), np.expand_dims(lung_image, 2), np.expand_dims(lung_image, 2)), 2)
        if self.transform_image:
            image, lung_image, _ = self.transform_image(images_formatted)

        # Correct lung images again to remove spurious background

        image = torch.unsqueeze(image, dim=0)
        
        # Change to a binary classification

        lung_mask[lung_mask != 0] = 1
        inf_mask[inf_mask != 0] = 1

        return image, lung_mask, inf_mask