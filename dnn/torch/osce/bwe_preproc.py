import os
import argparse
from typing import List

import numpy as np
from scipy import signal
from scipy.io import wavfile
import resampy


import math as m



parser = argparse.ArgumentParser()

parser.add_argument("filelist", type=str, help="file with filenames for concatenation in WAVE format")
parser.add_argument("target_fs", type=int, help="target sampling rate of concatenated file")
parser.add_argument("output", type=str, help="output directory")
parser.add_argument("--basedir", type=str, help="basedir for filenames in filelist, defaults to ./", default="./")
parser.add_argument("--normalize", action="store_true", help="apply normalization")
parser.add_argument("--db_max", type=float, help="max DB for random normalization", default=0)
parser.add_argument("--db_min", type=float, help="min DB for random normalization", default=0)
parser.add_argument("--random_eq_prob", type=float, help="portion of items to which random eq will be applied (default: 0.4)", default=0.4)
parser.add_argument("--static_noise_prob", type=float, help="portion of items to which static noise will be added (default: 0.2)", default=0.2)
parser.add_argument("--random_dc_prob", type=float, help="portion of items to which random dc offset will be added (default: 0.1)", default=0.1)
parser.add_argument("--rirdir", type=str, default=None, help="folder with room impulse responses in wav format (defaul: None)")
parser.add_argument("--rir_prob", type=float, default=0.0, help="portion of items to which a random rir is applied (default: 0)")
parser.add_argument("--verbose", action="store_true")

def read_filelist(basedir, filelist):
    with open(filelist, "r") as f:
        files = f.readlines()

    fullfiles = [os.path.join(basedir, f.rstrip('\n')) for f in files if len(f.rstrip('\n')) > 0]

    return fullfiles

def read_wave(file, target_fs):
    fs, x = wavfile.read(file)

    if fs < target_fs:
        return None
        print(f"[read_wave] warning: file {file} will be up-sampled from {fs} to {target_fs} Hz")

    if fs != target_fs:
        x = resampy.resample(x, fs, target_fs)

    return x.astype(np.float32)

def load_rirs(rirdir, target_fs):
    """ read rirs (assumed .wav) from subfolders of rirdir """

    rirs = []
    for dirpath, dirnames, filenames in os.walk(rirdir):
        for file in filenames:
            if file.endswith(".wav"):
                x = read_wave(os.path.join(dirpath, file), target_fs)
                x = x / np.max(np.abs(x))
                rirs.append(x)

    return rirs


lp_coeffs = signal.firwin(151, 20000, fs=48000)
def apply_20kHz_lp(x, fs):
    if fs != 48000:
        return x

    y = np.convolve(x, lp_coeffs, mode='valid')
    y *= np.max(np.abs(x)) / np.max(np.abs(y) + 1e-6)

    return y


def random_normalize(x, db_min, db_max, max_val=2**15 - 1):
    db = np.random.uniform(db_min, db_max, 1)
    m = np.abs(x).max()
    c = 10**(db/20) * max_val / m

    return c * x

def random_resamp16(x, fs=48000):
    assert fs == 48000 and "only supporting 48kHz input sampling rate for now"

    cutoff = 800 * np.random.rand(1) + 7200 # cutoff between 7.2 and 8 kHz
    numtaps = 2 * np.random.randint(75, 150) + 1
    a = signal.firwin(numtaps, cutoff, fs=fs)

    x16 = np.convolve(x, a, mode='same')[::3]

    return x16


def estimate_bandwidth(x, fs):
    assert fs >= 44100 and "currently only fs >= 44100 supported"
    f, t, X = signal.stft(x, nperseg=960, fs=fs)
    X = X.transpose()

    X_pow = np.abs(X) ** 2

    X_nrg = np.sum(X_pow, axis=1)
    threshold = np.sort(X_nrg)[int(0.9 * len(X_nrg))] * 0.1
    X_pow = X_pow[X_nrg > threshold]

    i = 0
    wb_nrg = 0
    wb_bands = 0
    while f[i] < 8000:
        wb_nrg += np.sum(X_pow[:, i])
        wb_bands += 1
        i += 1
    wb_nrg /= wb_bands

    i += 5 # safety margin
    swb_nrg = 0
    swb_bands = 0
    while f[i] < 16000:
        swb_nrg += np.sum(X_pow[:, i])
        swb_bands += 1
        i += 1
    swb_nrg /= swb_bands

    i += 5 # safety margin
    fb_nrg = 0
    fb_bands = 0
    while i < X_pow.shape[1]:
        fb_nrg += np.sum(X_pow[:, i])
        fb_bands += 1
        i += 1
    fb_nrg /= fb_bands


    if swb_nrg / wb_nrg < 1e-5:
        return 'wb'
    elif fb_nrg / wb_nrg < 1e-7:
        return 'swb'
    else:
        return 'fb'

def _get_random_eq_filter(num_taps=51, min_gain=1/3, max_gain=3, cutoff=8000, fs=48000, num_bands=15):

    nyquist = fs / 2
    freqs = (np.arange(num_bands)) / (num_bands - 1)
    cutoff = cutoff/nyquist
    log_min_gain = m.log(min_gain)
    log_max_gain = m.log(max_gain)
    split = int(cutoff * (num_bands - 1)) + 1


    log_gains =  np.random.rand(num_bands) * (log_max_gain - log_min_gain) + log_min_gain
    low_band_mean = np.mean(log_gains[:split])
    log_gains[:split] -= low_band_mean
    log_gains[split:] = 0
    gains = np.exp(log_gains)

    taps = signal.firwin2(num_taps, freqs, gains, nfreqs=127)


    return taps

def trim_silence(x, fs, threshold=0.005):
    frame_size = 320 * fs // 16000

    num_frames = len(x) // frame_size
    y = x[: frame_size * num_frames]

    frame_nrg = np.sum(y.reshape(-1, frame_size) ** 2, axis=1)
    ref_nrg = np.sort(frame_nrg)[int(num_frames * 0.9)]
    silence_threshold = threshold * ref_nrg

    for i, nrg in enumerate(frame_nrg):
        if nrg > silence_threshold:
            break

    first_active_frame_index = i

    for i in range(num_frames - 1, -1, -1):
        if frame_nrg[i] > silence_threshold:
            break

    last_active_frame_index = i

    i_start = max(first_active_frame_index - 20, 0) * frame_size
    i_stop = min(last_active_frame_index + 20, num_frames - 1) * frame_size

    return x[i_start:i_stop]



def random_eq(x, fs, cutoff):
    taps = _get_random_eq_filter(fs=fs, cutoff=cutoff)
    y = np.convolve(taps, x.astype(np.float32))

    # rescale
    y *= np.max(np.abs(x)) / np.max(np.abs(y + 1e-9))

    return y

def static_lowband_noise(x, fs, cutoff, max_gain=0.02):
    k_lp = (5 * fs // 16000)
    lp_taps = signal.firwin(2 * k_lp + 1, 2 * cutoff / fs)
    eq_taps = _get_random_eq_filter(num_bands=9)

    noise = np.random.randn(len(x) + len(lp_taps) + len(eq_taps) - 2)
    noise = np.convolve(noise, lp_taps, mode='valid')
    noise = np.convolve(noise, eq_taps, mode='valid')

    gain = np.random.rand(1) * max_gain

    x_max = np.max(np.abs(x))

    noise *= gain * x_max / np.max(np.abs(noise))

    y = x + noise
    y *= x_max / np.max(np.abs(y + 1e-9))

    return y

def apply_random_rir(x, rirs, rescale=True):
    i = np.random.randint(0, len(rirs))
    y = np.convolve(x, rirs[i], mode='same')
    if rescale: y *= np.max(np.abs(x)) / np.max(np.abs(y) + 1e-6)
    return y


def random_dc_offset(x, max_rel_offset=0.03):
    x_max = np.max(np.abs(x))
    offset = x_max * (2 * np.random.rand(1) - 1) * max_rel_offset

    y = x + offset
    y *= x_max / np.max(np.abs(y + 1e-9))

    return y


def concatenate(filelist : str,
                outdir : str,
                target_fs : int,
                normalize : bool=True,
                db_min : float=0,
                db_max : float=0,
                rand_eq_prob : float=0,
                static_noise_prob: float=0,
                rand_dc_prob : float=0,
                rirs : List = None,
                rir_prob : float = 0,
                verbose=False):

    overlap_size = int(40 * target_fs / 8000)
    overlap_mem = np.zeros(overlap_size, dtype=np.float32)
    overlap_win1 = (0.5 + 0.5 * np.cos(np.arange(0, overlap_size) * np.pi / overlap_size)).astype(np.float32)
    overlap_win2 = np.flipud(overlap_win1)

    # same for 16 kHz
    assert overlap_size % 3 == 0
    overlap_size16 = overlap_size // 3
    overlap_mem16 = np.zeros(overlap_size16, dtype=np.float32)
    overlap_win1_16 = overlap_win1[::3]
    overlap_win2_16 = np.flipud(overlap_win1_16)

    output48 = os.path.join(outdir, 'signal_48kHz.s16')
    output16 = os.path.join(outdir, 'signal_16kHz.s16')
    os.makedirs(outdir, exist_ok=True)

    with open(output48, 'wb') as f48, open(output16, 'wb') as f16:
        for file in filelist:
            x = read_wave(file, target_fs)
            if x is None: continue

            x = trim_silence(x, target_fs)

            x = apply_20kHz_lp(x, target_fs)

            bwidth = estimate_bandwidth(x, target_fs)
            if bwidth != 'fb':
                if verbose: print(f"bandwidth {bwidth} detected: skipping {file}...")
                continue

            if len(x) < 10 * overlap_size:
                if verbose: print(f"skipping {file}...")
                continue
            elif verbose:
                print(f"processing {file}...")

            noise_first = np.random.randint(2)

            if np.random.rand(1) < rand_eq_prob:
                x = random_eq(x, target_fs, 5000)

            if not noise_first:
                if np.random.rand(1) < rir_prob:
                    x = apply_random_rir(x, rirs)

            if np.random.rand(1) < static_noise_prob:
                x = static_lowband_noise(x, target_fs, 8000, max_gain=0.01)

            if noise_first:
                if np.random.rand(1) < rir_prob:
                    x = apply_random_rir(x, rirs)

            if np.random.rand(1) < rand_dc_prob:
                x = random_dc_offset(x)

            # trim final signal to length divisible by 3 to keep 16 and 48 kHz signals in sync
            x = x[:len(x) - (len(x) % 3)]

            if normalize:
                x = random_normalize(x, db_min, db_max)

            # write 48 and 16 kHz signals to disk
            if False:
                x1 = x[:-overlap_size]
                x1[:overlap_size] = overlap_win1 * overlap_mem + overlap_win2 * x1[:overlap_size]
                f48.write(x1.astype(np.int16).tobytes())

                x16 = random_resamp16(x)
                x1_16 = x16[:-overlap_size16]
                x1_16[:overlap_size16] = overlap_win1_16 * overlap_mem16 + overlap_win2_16 * x1_16[:overlap_size16]
                f16.write(x1_16.astype(np.int16).tobytes())

                # memory update
                overlap_mem = x[-overlap_size:]
                overlap_mem16 = x16[-overlap_size16:]
            else:
                # window and zero pad signal
                padding_samples = 3 * 100
                x[:overlap_size]  *= overlap_win2 # fade in
                x[-overlap_size:] *= overlap_win1 # fade out

                x = np.concatenate((np.zeros(padding_samples), x, np.zeros(padding_samples)), dtype=x.dtype)

                x16 = random_resamp16(x)

                assert 3*len(x16) == len(x)
                if np.max(x) > 2**15 - 1 or np.min(x) < -2**15: print("clipping")
                if np.max(x16) > 2**15 - 1 or np.min(x16) < -2**15: print("clipping")
                x = np.clip(x, -2**15, 2**15 - 1)
                x16 = np.clip(x16, -2**15, 2**15 - 1)
                f48.write(x.astype(np.int16).tobytes())
                f16.write(x16.astype(np.int16).tobytes())


if __name__ == "__main__":
    args = parser.parse_args()

    filelist = read_filelist(args.basedir, args.filelist)

    if args.rirdir is not None:
        rirs = load_rirs(args.rirdir, args.target_fs)
    else:
        rirs = []

    concatenate(filelist,
                args.output,
                args.target_fs,
                normalize=args.normalize,
                db_min=args.db_min,
                db_max=args.db_max,
                rand_eq_prob=args.random_eq_prob,
                static_noise_prob=args.static_noise_prob,
                rand_dc_prob=args.random_dc_prob,
                rirs=rirs,
                rir_prob=args.rir_prob,
                verbose=args.verbose)
