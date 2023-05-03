import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.image_loader import EnsembleLoader
import argparse
import numpy as np

def F1(prediction, target):
    recall = np.mean((prediction[target == 1] == 1).astype(np.float64))
    precision = np.mean((1 == target[prediction == 1]).astype(np.float64))
    return (2*precision*recall / (precision + recall + 1e-7))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp')
    parser.add_argument('--msk-file', type=str, default='')
    args = parser.parse_args()

    dataset = EnsembleLoader(exp=args.exp, msk_file=args.msk_file)
    logits = np.swapaxes(np.swapaxes(dataset.logits, 0, 1), 1, 2)
    target = dataset.masks.astype(np.int32)
    
    output = 1 / (1+np.exp(-logits.mean(axis=-1)))
    prediction = (output > 0.5).astype(np.int32)
    print(f"F-1 score (Averaging): {F1(prediction, target)}")

    output = 1 / (1+np.exp(-logits))
    prediction = (output > 0.5).astype(np.int32)
    prediction = (prediction.mean(axis=-1) > 0.5).astype(np.int32)
    print(f"F-1 score (Voting): {F1(prediction, target)}")
    

