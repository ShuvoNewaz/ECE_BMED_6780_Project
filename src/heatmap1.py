import argparse
import os

import numpy as np

import multiprocessing as mp

import nibabel as nib

import pickle

exp = "exp/heatmap"
masks = nib.load('data/val_mask.nii.gz').get_fdata().swapaxes(1, 2).swapaxes(0, 1)
thresholds = np.arange(0, 1, 0.04)

def load_logits(model_name):
    filename = os.path.join(exp, model_name, 'logits.npy')
    with open(filename, 'rb') as f:
        logits = np.load(f)
    return logits

def get_num_param(model_name):
    filename = os.path.join(exp, model_name, 'train.log')
    with open(filename, 'r') as f:
        nparam = [line for line in f.readlines() if "root INFO Number of parameters" in line][0]
    return float(nparam.split(" ")[-1])

def BinaryF1(logits, target, threshold):
    target = target.astype(np.int32)
    output = 1 / (1 + np.exp(-logits))
    prediction = (output > threshold).astype(np.int32)
    recall = np.mean((prediction[target == 1] == 1).astype(np.float32))
    precision = np.mean((1 == target[prediction == 1]).astype(np.float32))
    return np.nan_to_num((2*precision*recall / (precision + recall + 1e-10)), 0)

def getF1(logits):
    return [BinaryF1(logits, masks, threshold) for threshold in thresholds]

if __name__ == '__main__':
    models = os.listdir(exp)
    num_param = {model_name:get_num_param(model_name) for model_name in models}

    with mp.Pool() as pool:
        logits = pool.map(load_logits, models)
        F1 = {model: f1 for model, f1 in zip(models, pool.map(getF1, logits))}
    F1["thresholds"] = thresholds
    F1["num_param"] = num_param

    with open("heatmap1.pkl", "wb") as f:
        pickle.dump(F1, f)
    