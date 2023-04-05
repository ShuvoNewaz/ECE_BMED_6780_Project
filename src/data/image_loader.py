import os
import torch
from torchvision.transforms import Compose
from torch.utils import data
import nibabel as nib
import numpy as np


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
