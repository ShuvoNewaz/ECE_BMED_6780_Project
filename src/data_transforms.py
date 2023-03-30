from torchvision import transforms


def get_train_transforms() -> transforms.Compose:
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.RandomHorizontalFlip(0.6), # Comment
                                            # transforms.RandomVerticalFlip(0.6),
                                            # transforms.GaussianBlur([5, 1]),
                                            # transforms.RandomCrop(480), # Comment
                                            # transforms.RandomRotation(30), # Comment to obtain required accuracy in ResNets
                                            # transforms.Normalize(mean=mean, std=std)
                                        ])

    return data_transforms


def get_val_transforms() -> transforms.Compose:
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]
    data_transforms = transforms.Compose([
                                            transforms.ToTensor(),
                                            # transforms.Normalize(mean=mean, std=std)
                                        ])

    return data_transforms