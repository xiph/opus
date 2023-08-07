"""
Perform Data Augmentation (Gain, Additive Noise, Random Filtering) on Input TTS Data
1. Read in chunks and compute clean pitch first
2. Then add in augmentation (Noise/Level/Response)
    - Adds filtered noise from the "Demand" dataset, https://zenodo.org/record/1227121#.XRKKxYhKiUk
    - When using the Demand Dataset, consider each channel as a possible noise input, and keep the first 4 minutes of noise for training
3. Use this "augmented" audio for feature computation, and compute pitch using CREPE on the clean input

Notes: To ensure consistency with the discovered CREPE offset, we do the following
- We pad the input audio to the zero-centered CREPE estimator with 80 zeros
- We pad the input audio to our feature computation with 160 zeros to center them
"""

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('data', type=str, help='input raw audio data')
parser.add_argument('output', type=str, help='output IF features')
parser.add_argument('crepe_pitch', type=str, help='.npy file output containing target Pitch')
parser.add_argument('noise_dataset', type=str, help='Location of the Demand Datset')
parser.add_argument('--fraction_input_use', type=float, help='Fraction of input data to consider',default = 0.3,required = False)
parser.add_argument('--gpu_index', type=int, help='GPU index to use if multiple GPUs',default = 0,required = False)
parser.add_argument('--choice_augment', type=str, help='Choice of noise augmentation, either use additive synthetic noise or add noise from the demand dataset',choices = ['demand','synthetic'],default = "demand",required = False)
parser.add_argument('--fraction_clean', type=float, help='Fraction of data to keep clean (that is not augment with anything)',default = 0.2,required = False)
parser.add_argument('--chunk_size', type=int, help='Number of samples to augment with for each iteration',default = 16000,required = False)
parser.add_argument('--N', type=int, help='STFT window size',default = 320,required = False)
parser.add_argument('--H', type=int, help='STFT Hop size',default = 160,required = False)
parser.add_argument('--freq_keep', type=int, help='Number of Frequencies to keep',default = 30,required = False)

args = parser.parse_args()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

from utils import stft, random_filter

import numpy as np
import tqdm
import crepe
import random
import glob

data_full = np.memmap(args.data, dtype=np.int16,mode = 'r')
data = data_full[:(int)(args.fraction_input_use*data_full.shape[0])]

# list_features = []
list_cents = []
list_confidences = []

N = args.N
H = args.H
freq_keep = args.freq_keep
# Minimum/Maximum periods, decided by LPCNet
min_period = 32
max_period = 256
f_ref = 16000/max_period
chunk_size = args.chunk_size
num_frames_chunk = chunk_size//H
list_indices_keep = np.concatenate([np.arange(freq_keep), (N//2 + 1) + np.arange(freq_keep), 2*(N//2 + 1) + np.arange(freq_keep)])

output  = np.memmap(args.output, dtype=np.float32, shape=(((data.shape[0]//chunk_size - 1)//1)*num_frames_chunk,list_indices_keep.shape[0]), mode='w+')

fraction_clean = args.fraction_clean

noise_dataset = args.noise_dataset
list_nfiles = ['DKITCHEN','NFIELD','OHALLWAY','PCAFETER','SPSQUARE','TCAR','DLIVING','NPARK','OMEETING','PRESTO','STRAFFIC','TMETRO','DWASHING','NRIVER','OOFFICE','PSTATION','TBUS']

for i in tqdm.trange((data.shape[0]//chunk_size - 1)//1):
    chunk = data[i*chunk_size:(i + 1)*chunk_size]/(2**15 - 1)

    # Clean Pitch/Confidence Estimate
    # Padding input to CREPE by 80 samples to ensure it aligns
    _, pitch, confidence, _ = crepe.predict(np.concatenate([np.zeros(80),chunk]), 16000, center=True, viterbi=True,verbose=0)
    cent = 1200*np.log2(np.divide(pitch, f_ref, out=np.zeros_like(pitch), where=pitch!=0) + 1.0e-8)

    # Filter out of range pitches/confidences
    confidence[pitch < 16000/max_period] = 0
    confidence[pitch > 16000/min_period] = 0

    # Keep fraction of data clean, augment only 1 minus the fraction
    if (np.random.rand() > fraction_clean):
        # Response, generate controlled/random 2nd order IIR filter and filter chunk
        chunk = random_filter(chunk)

        # Level/Gain response {scale by random gain between 1.0e-3 and 10}
        # Generate random gain in dB and then convert to scale
        g_dB = np.random.uniform(low =  -60, high = 20, size = 1)
        # g_dB = 0
        g = 10**(g_dB/20)

        # Noise Addition {Add random SNR 2nd order randomly colored noise}
        # Generate noise SNR value and add corresponding noise
        snr_dB = np.random.uniform(low =  -20, high = 30, size = 1)

        if args.choice_augment == 'synthetic':
            n = np.random.randn(chunk_size)
        else:
            list_noisefiles = noise_dataset + random.choice(list_nfiles) + '/ch*.wav'
            noise_file = random.choice(glob.glob(list_noisefiles))
            n = np.memmap(noise_file, dtype=np.int16)/(2**15 - 1)
            rand_range = np.random.randint(low = 0, high = (n.shape[0] - 16000 - chunk.shape[0])//2) # 16000 is subtracted because we will use the last 16000 minutes of noise for testing
            n = n[rand_range:rand_range + chunk.shape[0]]
        
        # Randomly filter the sampled noise as well
        n = random_filter(n)
        snr_multiplier = np.sqrt((np.sum(np.abs(chunk)**2)/np.sum(np.abs(n)**2))*10**(-snr_dB/10))

        chunk = g*(chunk + snr_multiplier*n)

    # Zero pad input audio by 160 to center the frames
    spec = stft(x = np.concatenate([np.zeros(160),chunk]), w = 'boxcar', N = N, H = H).T
    phase_diff = spec*np.conj(np.roll(spec,1,axis = -1))
    phase_diff = phase_diff/(np.abs(phase_diff) + 1.0e-8)
    feature = np.concatenate([np.log(np.abs(spec) + 1.0e-8),np.real(phase_diff),np.imag(phase_diff)],axis = 0).T
    feature = feature[:,list_indices_keep]

    num_frames = min(cent.shape[0],feature.shape[0],num_frames_chunk)
    feature = feature[:num_frames,:]
    cent = cent[:num_frames]
    confidence = confidence[:num_frames]
    output[i*num_frames_chunk:(i + 1)*num_frames_chunk,:] = feature
    list_cents.append(cent)
    list_confidences.append(confidence)

list_cents = np.hstack(list_cents)
list_confidences = np.hstack(list_confidences)

np.save(args.crepe_pitch,np.vstack([list_cents,list_confidences]))








