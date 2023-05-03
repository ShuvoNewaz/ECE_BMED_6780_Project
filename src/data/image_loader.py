import os
import torch
from torchvision.transforms import Compose
from torch.utils import data
import nibabel as nib
import numpy as np
from typing import List, Tuple

import multiprocessing as mp

class ImageLoader(data.Dataset):
    def __init__(self, im_file: str, msk_file: str, transform: Compose=None) -> None:
        """
        args:
            root_dir: Root working directory
            split: 
            im_file: Path to image_file.nii.gz
            msk_file: Path to mask_file.nii.gz
            seg_type: Lung or Infection Segmentation
        """
        super().__init__()
        # self.root_dir = root_dir
        # self.data_dir = os.path.join(root_dir, 'data')
        self.im_file = im_file
        self.msk_file = msk_file
        self.transform = transform
        self.dataset = self.load_images_with_masks()

    def load_images_with_masks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the list of tuples containing the image and the mask
        of the dataset.
        """
        dataset = []
        images = nib.load(self.im_file).get_fdata()
        if not self.msk_file:
            masks = np.zeros(images.shape) # Placeholder for missing validation masks
        else:
            masks = nib.load(self.msk_file).get_fdata()
        for i in range(images.shape[2]):
            dataset.append((images[:, :, i], masks[:, :, i]))

        return dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        if self.transform:
            image = self.transform(image)

        return image, mask

class EnsembleLoader(data.Dataset):
    def __init__(self, exp: str, msk_file: str, logit_file="logits.npy", 
                    transform: Compose=None) -> None:
        """
        args:
            root_dir: Root working directory
            split: 
            im_file: Path to image_file.nii.gz
            msk_file: Path to mask_file.nii.gz
            seg_type: Lung or Infection Segmentation
        """
        super().__init__()
        # self.root_dir = root_dir
        # self.data_dir = os.path.join(root_dir, 'data')
        self.exp = exp
        self.msk_file = msk_file
        self.transform = transform
        self.logit_file = logit_file
        self.load_images_with_masks()
    def load_npy(self, fileName):
        # with open(fileName, 'rb') as f:
        result = np.load(fileName, mmap_mode='r')
        return np.expand_dims(result, axis=-1)
    def load_images_with_masks(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns the list of tuples containing the image and the mask
        of the dataset.
        """
        logit_npys = [os.path.join(self.exp, i, self.logit_file) for i in sorted(os.listdir(self.exp))]
        logit_npys = [i for i in logit_npys if os.path.isfile(i)]
        logits = [self.load_npy(i) for i in logit_npys]
        if not self.msk_file:
            masks = np.zeros(images.shape) # Placeholder for missing validation masks
        else:
            masks = nib.load(self.msk_file).get_fdata()

        self.logits = logits # [(N, H, W, 1)]
        self.masks = masks

    def __len__(self):
        return self.logits[0].shape[0]
        # return len(self.dataset)

    def __getitem__(self, index):
        # image, mask = self.dataset[index]
        mask = self.masks[:, :, index]
        image = np.concatenate([i[index] for i in self.logits], axis=-1)
        image = image
        if self.transform:
            image = self.transform(image)

        return image, mask
