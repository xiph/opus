import argparse
import sys
sys.path.append('./')

import torch
from utils.bwe_features import load_inference_data
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('testsignal', type=str)

if __name__ == "__main__":
    args = parser.parse_args()

    _, features = load_inference_data(args.testsignal)

    N = features.shape[0]

    for n in range(N):
        print(f"frame[{n}]")
        print(f"lmspec: {features[n, :32]}")
        print(f"freqs: {features[n,32:]}")
