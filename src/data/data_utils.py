import numpy as np


def extract_lung_regions(image, lung_mask, lung_image):
    lung_index = lung_mask != 0
    background_index = lung_mask == 0
    lung_image[lung_index] = image[lung_index]
    lung_image[background_index] = image.min()

    return lung_image