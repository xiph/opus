import argparse
import sys
sys.path.append('./')

import torch
from utils.spec import create_filter_bank
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str)



if __name__ == "__main__":
    args = parser.parse_args()

    c = torch.load(args.checkpoint, map_location='cpu')

    num_bands = c['setup']['data']['spec_num_bands']
    fb, center_bins = create_filter_bank(num_bands, n_fft=320, fs=16000, scale='erb', round_center_bins=True, normalize=False, return_center_bins=True)
    weights = 1/fb.sum(axis=-1)

    print(f"center_bins:")

    print("".join([f"{int(cb):4d}," for cb in center_bins]))

    print(f"band_weights:")
    print("".join([f" {w:1.9f}," for w in weights]))