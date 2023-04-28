import os
import torch
from torchvision import transforms
from torchvision.transforms import Compose
from torch.utils import data
import nibabel as nib
import numpy as np


class ImageLoader(data.Dataset):
    def __init__(self, split: str, im_file: str, lung_msk_file: str, inf_msk_file: str, transform_common: Compose=None, transform_image: Compose=None) -> None:
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
        self.split = split
        self.im_file = im_file
        self.lung_msk_file = lung_msk_file
        self.inf_msk_file = inf_msk_file
        self.transform_common = transform_common
        self.transform_image = transform_image
        self.dataset = self.load_images_with_masks()

    def load_images_with_masks(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Returns the list of tuples containing the image and the mask
        of the dataset.
        """
        dataset = []
        images = nib.load(self.im_file).get_fdata()
        # if self.split == 'validation' or self.split == 'test' or self.split == 'val':
        #     lung_masks = np.zeros(images.shape) # Placeholder for missing validation masks
        # else:
        if self.lung_msk_file is not None:
            lung_masks = nib.load(self.lung_msk_file).get_fdata()
        else:
            lung_masks = np.zeros(images.shape) # Placeholder for missing validation masks
        inf_masks = nib.load(self.inf_msk_file).get_fdata()
        for i in range(images.shape[2]):
            dataset.append((images[:, :, i], lung_masks[:, :, i], inf_masks[:, :, i]))

        return dataset

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):
        image, lung_mask, inf_mask = self.dataset[index]
        
        lung_index = lung_mask != 0
        background_index = lung_mask == 0
        
        lung_image = np.zeros(image.shape)
        lung_image[lung_index] = image[lung_index]
        lung_image[background_index] = np.min(image)

        # Facilitate transformation of masks

        images_formatted = np.concatenate((np.expand_dims(image, 2), np.expand_dims(lung_image, 2)), 2)
        image_and_mask = np.concatenate((images_formatted, np.expand_dims(lung_mask, 2), np.expand_dims(inf_mask, 2)), 2)

        # if self.split == 'train':
        # Add rotation and flips

        if self.transform_common:
            image, lung_image, lung_mask, inf_mask = self.transform_common(image_and_mask)
        
        # Add noise and jitters

        images_formatted = np.concatenate((np.expand_dims(image, 2), np.expand_dims(lung_image, 2), np.expand_dims(lung_image, 2)), 2)
        if self.transform_image:
            image, lung_image, _ = self.transform_image(images_formatted)

        # Correct lung images again to remove spurious background

        lung_index = lung_mask != 0
        background_index = lung_mask == 0
        lung_image[lung_index] = image[lung_index]
        lung_image[background_index] = torch.min(image).item()

        image = torch.unsqueeze(image, dim=0)
        lung_image = torch.unsqueeze(lung_image, dim=0)
        
        # Change to a binary classification

        lung_mask[lung_mask != 0] = 1
        inf_mask[inf_mask != 0] = 1

        # if self.split == 'validation' or self.split == 'test' or self.split == 'val':
        #     return image, inf_mask
        # else:
        return image, lung_image, lung_mask, inf_mask


class ImageLoaderFile(data.Dataset):
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

    def load_images_with_masks(self) -> list[tuple[np.ndarray, np.ndarray]]:
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
    

class ImageLoaderTensor(data.Dataset):
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

    def load_images_with_masks(self) -> list[tuple[np.ndarray, np.ndarray]]:
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
