import os
import argparse

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

parser = argparse.ArgumentParser()
parser.add_argument('ref_wav', type=str, help='reference wav file')
parser.add_argument('test_wav', type=str, help='test wav file')

opus_eband5ms = [0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100]
opus_eband10ms = [2 * b for b in opus_eband5ms]

def create_filter_bank(band_limits, num_bins, normalize=False):
    filters = []
    for i in range(len(band_limits) - 1):
        i_start, i_stop = band_limits[i], band_limits[i + 1]
        coeffs = np.zeros(num_bins)
        coeffs[i_start:i_stop] = 1

        if normalize:
            coeffs = coeffs / sum(coeffs)

        filters.append(coeffs.reshape(1, -1))

    fb = np.concatenate(filters, axis=0)
    return fb

fb = create_filter_bank(opus_eband10ms, 241)

highband_idx = -4
func = lambda x : x ** 0.25

def band_wise_distortion(Xref, Xcut, p=None):
    Xref = np.abs(Xref)**2
    Xcut = np.abs(Xcut)**2

    Yref = fb @ Xref
    Ycut = fb @ Xcut

    nf = np.max(Yref, axis=1) * (10**(-30/10))
    
    Yref = func(np.maximum(Yref, nf.reshape(-1, 1)))
    Ycut = func(np.maximum(Ycut, nf.reshape(-1, 1)))
    delta = np.abs(Yref - Ycut)

    if p is not None:
        delta = np.linalg.norm(delta / delta.shape[-1], ord=p, axis=1)
    
    return 1000 * delta[highband_idx:]


def main(ref_path, cut_path):
    args = parser.parse_args()
    # Read WAV files
    _, x_ref = wavfile.read(ref_path)
    _, x_cut = wavfile.read(cut_path)

    m = np.max(np.abs(x_ref))
    x_ref = x_ref / m
    x_cut = x_cut / m
    
    # Compute STFTs
    _, _, X_ref = stft(x_ref, nperseg=480)
    _, _, X_cut = stft(x_cut, nperseg=480)
    
    # lowpass reference
    X_lp = X_ref.copy()
    X_lp[80:, :] = 0
    
    # flip comparison
    X_flip = X_ref.copy()
    X_flip[80:, :] = X_ref[80:, ::-1]
    
    # Calculate distortions
    dist_ref_cut = band_wise_distortion(X_ref, X_cut, 2)
    dist_ref_lp = band_wise_distortion(X_ref, X_lp, 2)
    dist_ref_flip = band_wise_distortion(X_ref, X_flip, 2)
    
    # Print results
    ref_cut_pass  = all(dist_ref_cut <= dist_ref_lp)
    ref_flip_pass = all(dist_ref_flip <= dist_ref_lp)
    print(f"ref-test distortion: {dist_ref_cut} ({'pass' if ref_cut_pass else 'fail'})")
    print(f"ref-lp   distortion: {dist_ref_lp}")
    print(f"ref-flip distortion: {dist_ref_flip} ({'pass' if ref_flip_pass else 'fail'})")
    
if __name__ == "__main__":
    # Example usage: update these paths and filename as needed
    args = parser.parse_args()
    main(args.ref_wav, args.test_wav)