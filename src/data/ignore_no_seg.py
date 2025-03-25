import numpy as np
import nibabel as nib
from argparse import ArgumentParser


parser = ArgumentParser(description='Removes all images that do not have a lung mask')
parser.add_argument('--data_directory', '-dir', type=str, required=True, help="string representing the directory to read the files")
args = parser.parse_args()

datadir = args.data_directory

images = nib.load(f'{datadir}/im.nii.gz').get_fdata()
lung_mask = nib.load(f'{datadir}/lung_mask.nii.gz').get_fdata()
mask = nib.load(f'{datadir}/mask.nii.gz').get_fdata()

# Extract lung regions using lung segmentation masks

seg_count = 0
total_images = images.shape[2]
lung_regions = np.zeros(images.shape)
for i in range(total_images):
    lung_index = lung_mask[:, :, i] != 0
    ignore_index = lung_mask[:, :, i] == 0
    lung_regions[:, :, i][lung_index] = images[:, :, i][lung_index]
    lung_regions[:, :, i][ignore_index] = np.min(images[:, :, i][ignore_index])
    seg_count += len(np.unique(mask[:, :, i])) > 1

# Save the extracted lung regions

nifti_lung_region = nib.Nifti1Image(lung_regions, affine=np.eye(4))

# Remove unsegmented images and masks

print("Removing data that does not have a mask.")

include_image, include_mask, = np.zeros((images.shape[0], images.shape[1], seg_count)), np.zeros((images.shape[0], images.shape[1], seg_count))
include_lung_image, include_lung_mask = np.zeros((images.shape[0], images.shape[1], seg_count)), np.zeros((images.shape[0], images.shape[1], seg_count))

count = 0
for i in range(total_images):
    if len(np.unique(mask[:, :, i])) > 1:
        include_image[:, :, count] = images[:, :, i]
        include_mask[:, :, count] = mask[:, :, i]
        include_lung_image[:, :, count] = lung_regions[:, :, i]
        include_lung_mask[:, :, count] = lung_mask[:, :, i]
        count += 1

# Save new .gz files

nifti_im = nib.Nifti1Image(include_image, affine=np.eye(4))
nifti_msk = nib.Nifti1Image(include_mask, affine=np.eye(4))
nifti_lung_im = nib.Nifti1Image(include_lung_image, affine=np.eye(4))
nifti_lung_msk = nib.Nifti1Image(include_lung_mask, affine=np.eye(4))

print("Saving new images and masks.")

nib.save(nifti_im, f'{datadir}/im.nii.gz')
nib.save(nifti_msk, f'{datadir}/mask.nii.gz')
nib.save(nifti_lung_im, f'{datadir}/lung_im.nii.gz')
nib.save(nifti_lung_msk, f'{datadir}/lung_mask.nii.gz')

