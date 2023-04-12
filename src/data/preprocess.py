import numpy as np
import os
import nibabel as nib
import torch
from torchvision import transforms
import torchio as tio
from argparse import ArgumentParser


resize = transforms.Compose([transforms.Resize([512, 512])])
parser = ArgumentParser(description='Creates a collection of nifti images files from distributed sources')
parser.add_argument('--read_directory', '-rd', type=str, required=True, help="string representing the directory to read the files")
parser.add_argument('--save_directory', '-sd', type=str, required=True, help="directory+name with which the concatenated image/mask will be")
parser.add_argument('--data_type', '-dt', type=str, choices=['image', 'mask'], required=True, help="whether image or mask is being transformed")
args = parser.parse_args()

read_dir = args.read_directory
save_dir = args.save_directory
data_type = args.data_type

nii_list = []
for im_dir in os.listdir(read_dir):
    name_split = im_dir.split('.')
    if name_split[len(name_split) - 1] == 'gz':
        nii_list.append(os.path.join(read_dir, im_dir))
nii_list.sort()
for count, im_dir in enumerate(nii_list):
    print(f"Loading {im_dir}")
    if data_type == 'image':
        nifti_image = tio.ScalarImage(im_dir)   # Read the *.nii image
    elif data_type == 'mask':
        nifti_image = tio.LabelMap(im_dir)   # Read the *.nii mask
    else: raise NotImplementedError
    torch_image = nifti_image.data          # Extract the tensor data
    torch_image = torch.squeeze(torch_image, dim=0)
    torch_image = torch.swapaxes(torch_image, 0, 2)
    torch_image = resize(torch_image)
    torch_image = torch.swapaxes(torch_image, 0, 2)
    if count == 0:
        im_concat = torch_image
    else:
        im_concat = torch.concat((im_concat, torch_image), dim=2)
nifti_im_concat = nib.Nifti1Image(im_concat.numpy(), affine=np.eye(4))
nib.save(nifti_im_concat, save_dir)

    # return im_concat
