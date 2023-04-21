from typing import Tuple, List
import torch
import torch.nn.functional as F
import numpy as np


def Intersection_Union(
    output: torch.Tensor, target: torch.Tensor, K: int, ignore_index: int = 255) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes IOU in pure Pytorch.
    Note output and target sizes are N or N * L or N * H * W
    Args:
        output: Pytorch tensor represeting predicted label map,
            each value in range 0 to K - 1.
        target: Pytorch tensor representing ground truth label map,
            each value in range 0 to K - 1.
        K: integer number of possible classes
        ignore_index: integer representing class index to ignore
        cuda_available: CUDA is available to Pytorch to use
    Returns:
        area_intersection: 1d Pytorch tensor of length (K,) with counts
            for each of K classes, where pred & target matched
        area_union: 1d Pytorch tensor of length (K,) with counts
        area_target: 1d Pytorch tensor of length (K,) with bin counts
            for each of K classes, present in this GT label map.
    """
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K - 1)
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K - 1)
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection

    return area_intersection, area_union, area_target


def IOU(prediction, target):
    prediction = prediction.numpy()
    target = target.numpy()
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score


def BinaryF1(prediction, target):
    recall = torch.mean((prediction[target == 1] == 1).float())
    precision = torch.mean((1 == target[prediction == 1]).float())

    return (2*precision*recall / (precision + recall)).nan_to_num(0)


def ange_structure_loss(pred, mask, smooth=1):
    mask = mask.float()
    
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')
    # print(weit.shape, wbce)
    wbce = (weit*wbce).sum(dim=(1, 2)) / weit.sum(dim=(1, 2))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(1, 2))
    union = ((pred + mask)*weit).sum(dim=(1, 2))
    wiou = 1 - (inter + smooth)/(union - inter + smooth)
    
    return (wbce + wiou).mean()


def dice_loss_coff(pred, target, smooth = 0.0001):
    
    num = target.size(0)
    pred = pred.contiguous()
    target = target.contiguous()    

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return loss.sum()/num


def sample_variance(y: List[np.ndarray]):
    """
        Args:
            y: Probability of classes of all test images in all neural nets (T, no. of test images, C, H, W)
        Returns:
            var: Sample Variance (no. of test images, H, W)
    """
    T = len(y)
    N, C, H, W = y[0].shape
    var = 0
    for c in range(C):
        E_x2, E_2x = 0, 0
        for t in range(T):
            E_x2 = E_x2 + (y[t][:, c, :, :])**2
            E_2x = E_2x + y[t][:, c, :, :]
        E_x2 = 1/T * E_x2
        E_2x = (1/T * E_2x) ** 2
        var = var + E_x2 - E_2x
    var = 1/C * var

    return var


def predictive_entropy(y: List[np.ndarray]):
    """
        Args:
            C: Number of classes
            y: Probability of classes of all test images in all neural nets (T, no. of test images, C, H, W)
        Returns:
            H: Predictive Entropy (no. of test images, H, W)
    """
    T = len(y)
    N, C, H, W = y[0].shape
    H = 0
    for c in range(C):
        term = 0
        for t in range(T):
            term = term + y[t][:, c, :, :]
        term = 1/T * term
        H = H + term * torch.log(term)
    H = - H

    return H


def mutual_information(y: List[np.ndarray]):
    """
        Args:
            y: Probability of classes of all test images in all neural nets (T, no. of test images, C, H, W)
        Returns:
            MI: Mutual Information (no. of test images, C, H, W)
    """
    T = len(y)
    N, C, H, W = y[0].shape
    MI = np.zeros((N, C, H, W))

    H = predictive_entropy(y)
    for c in range(C):
        term = 0
        for t in range(T):
            entropy = y[t][:, c, :, :] * np.log(y[t][:, c, :, :])
            term = term + entropy
        term = -1/T * term
        MI[:, c, :, :] = H - term
    
    return MI


