"""
/* Copyright (c) 2023 Amazon
   Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""
import sys

import numpy as np
import scipy.signal

def compute_vad_mask(x, fs, stop_db=-70):

    frame_length = (fs + 49) // 50
    x = x[: frame_length * (len(x) // frame_length)]

    frames = x.reshape(-1, frame_length)
    frame_energy = np.sum(frames ** 2, axis=1)
    frame_energy_smooth = np.convolve(frame_energy, np.ones(5) / 5, mode='same')

    max_threshold = frame_energy.max() * 10 ** (stop_db/20)
    vactive = np.ones_like(frames)
    vactive[frame_energy_smooth < max_threshold, :] = 0
    vactive = vactive.reshape(-1)

    filter = np.sin(np.arange(frame_length) * np.pi / (frame_length - 1))
    filter = filter / filter.sum()

    mask = np.convolve(vactive, filter, mode='same')

    return x, mask

def convert_mask(mask, num_frames, frame_size=160, hop_size=40):
    num_samples = frame_size + (num_frames - 1) * hop_size
    if len(mask) < num_samples:
        mask = np.concatenate((mask, np.zeros(num_samples - len(mask))), dtype=mask.dtype)
    else:
        mask = mask[:num_samples]

    new_mask = np.array([np.mean(mask[i*hop_size : i*hop_size + frame_size]) for i in range(num_frames)])

    return new_mask

def power_spectrum(x, window_size=160, hop_size=40, window='hamming'):
    num_spectra = (len(x) - window_size - hop_size) // hop_size
    window = scipy.signal.get_window(window, window_size)
    N = window_size // 2

    frames = np.concatenate([x[np.newaxis, i * hop_size : i * hop_size + window_size] for i in range(num_spectra)]) * window
    psd = np.abs(np.fft.fft(frames, axis=1)[:, :N + 1]) ** 2

    return psd


def frequency_mask(num_bands, up_factor, down_factor):

    up_mask = np.zeros((num_bands, num_bands))
    down_mask = np.zeros((num_bands, num_bands))

    for i in range(num_bands):
        up_mask[i, : i + 1] = up_factor ** np.arange(i, -1, -1)
        down_mask[i, i :] = down_factor ** np.arange(num_bands - i)

    return down_mask @ up_mask


def rect_fb(band_limits, num_bins=None):
    num_bands = len(band_limits) - 1
    if num_bins is None:
        num_bins = band_limits[-1]

    fb = np.zeros((num_bands, num_bins))
    for i in range(num_bands):
        fb[i, band_limits[i]:band_limits[i+1]] = 1

    return fb


# base analysis grid: 10 ms window, 2.5 ms hop at 16 kHz -> 100 Hz bins spanning 0..8 kHz
BASE_RATE = 16000
BASE_WINDOW = 160
BASE_HOP = 40
LOWBAND_BINS = BASE_WINDOW // 2 + 1   # 81 bins covering 0..8 kHz on a 100 Hz grid


def lowband_psd(sig, fs):
    """ Short-time power spectrum of sig restricted to the 0..8 kHz lowband.

    Resampling to the 16 kHz lowband is done in the STFT domain: a fixed 10 ms
    window and 2.5 ms hop (scaled by the integer rate factor fs / 16 kHz) are
    used so that the first LOWBAND_BINS DFT bins always land on the same 100 Hz
    frequency grid regardless of fs. Keeping those bins is equivalent to an
    ideal-brickwall downsample to 16 kHz for the purpose of this metric (and the
    higher bins remain available for a future highband evaluation). The
    magnitude is normalised by factor**2 so the result is on the same scale as a
    native 16 kHz (factor 1) analysis, keeping the fs == 16 kHz path identical to
    the original metric.
    """
    if fs % BASE_RATE != 0:
        raise ValueError(f"sampling rate {fs} is not a multiple of {BASE_RATE}")
    factor = fs // BASE_RATE
    psd = power_spectrum(sig, window_size=BASE_WINDOW * factor, hop_size=BASE_HOP * factor)
    return psd[:, :LOWBAND_BINS] / (factor ** 2)


def _compare(x, y, fs_x=16000, fs_y=16000, apply_vad=False):
    """ Modified version of opus_compare, evaluating the 0..8 kHz lowband.

    x and y may be sampled at different integer multiples of 16 kHz; both are
    reduced to the common 16 kHz lowband representation in the STFT domain
    before the comparison, so an extending (e.g. 48 kHz) decoder output can be
    scored against a 16 kHz reference.

    Args:
        x (np.ndarray): reference input signal scaled to [-1, 1]
        y (np.ndarray): test signal scaled to [-1, 1]
        fs_x (int): sampling rate of x (multiple of 16000)
        fs_y (int): sampling rate of y (multiple of 16000)

    Returns:
        float: perceptually weighted error
    """
    # filter bank: bark scale with minimum-2-bin bands and cutoff at 7.5 kHz
    band_limits = [0, 2, 4, 6, 7, 9, 11, 13, 15, 18, 22, 26, 31, 36, 43, 51, 60, 75]
    num_bands = len(band_limits) - 1
    fb = rect_fb(band_limits, num_bins=LOWBAND_BINS)

    # trim to common duration (rates may differ, so align in time not samples)
    duration = min(len(x) / fs_x, len(y) / fs_y)
    x = x[:int(duration * fs_x)].copy() * 2**15
    y = y[:int(duration * fs_y)].copy() * 2**15

    psd_x = lowband_psd(x, fs_x) + 100000
    psd_y = lowband_psd(y, fs_y) + 100000

    # hop is a fixed 2.5 ms at every rate, so frame counts match up to rounding
    num_frames = min(psd_x.shape[0], psd_y.shape[0])
    psd_x = psd_x[:num_frames]
    psd_y = psd_y[:num_frames]

    # average band energies
    be_x = (psd_x @ fb.T) / np.sum(fb, axis=1)

    # frequecy masking
    f_mask = frequency_mask(num_bands, 0.1, 0.03)
    mask_x = be_x @ f_mask.T

    # temporal masking (2.5 ms per frame independent of rate)
    for i in range(1, num_frames):
        mask_x[i, :] += 0.5 * mask_x[i-1, :]

    # apply mask
    masked_psd_x = psd_x + 0.1 * (mask_x @ fb)
    masked_psd_y = psd_y + 0.1 * (mask_x @ fb)

    # 2-frame average
    masked_psd_x = masked_psd_x[1:] +  masked_psd_x[:-1]
    masked_psd_y = masked_psd_y[1:] +  masked_psd_y[:-1]

    # distortion metric
    re = masked_psd_y / masked_psd_x
    #im = re - np.log(re) - 1
    im = np.log(re) ** 2
    Eb = ((im @ fb.T) / np.sum(fb, axis=1))
    Ef = np.mean(Eb ** 1, axis=1)

    if apply_vad:
        factor_x = fs_x // BASE_RATE
        _, mask = compute_vad_mask(x, fs_x)
        mask = convert_mask(mask, Ef.shape[0], frame_size=BASE_WINDOW * factor_x, hop_size=BASE_HOP * factor_x)
    else:
        mask = np.ones_like(Ef)

    tmp = np.abs(Ef) ** 3
    tmp = np.convolve(tmp, np.ones(400) / 400, mode='same')
    err = np.max(tmp[mask > 1e-6]) ** (1/6)

    return float(err)

def compare(x, y, fs_x=16000, fs_y=16000, apply_vad=False):
    err = np.linalg.norm([_compare(x, y, fs_x=fs_x, fs_y=fs_y, apply_vad=apply_vad)], ord=2)
    return err


# ---------------------------------------------------------------------------
# Highband (bandwidth-extension) evaluation
# ---------------------------------------------------------------------------
# Opus eband filter bank on the 5 ms grid and its 10 ms version (100 Hz/bin at
# 48 kHz with nperseg=480). Ported from evaluation/highband_eval.py so that moc
# is the single source of truth for both the lowband and highband metrics.
opus_eband5ms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100]
opus_eband10ms = [2 * b for b in opus_eband5ms]

HIGHBAND_NPERSEG = 480                             # 48000 / 480 = 100 Hz per bin
HIGHBAND_NUM_BINS = HIGHBAND_NPERSEG // 2 + 1      # 241
HIGHBAND_NUM_BANDS = 4                             # last 4 ebands: 8-9.6, 9.6-12, 12-15.6, 15.6-20 kHz
HIGHBAND_LP_BIN = 80                               # bins >= 80 (>= 8 kHz) form the highband
_highband_fb = rect_fb(opus_eband10ms, num_bins=HIGHBAND_NUM_BINS)


def _highband_band_distortion(X_ref, X_test):
    """ Per-highband distortion between two 48 kHz STFTs (see highband_eval.py).

    Returns the distortion for the last HIGHBAND_NUM_BANDS eband groups
    (8.0-9.6, 9.6-12.0, 12.0-15.6, 15.6-20.0 kHz).
    """
    Pref = np.abs(X_ref) ** 2
    Ptest = np.abs(X_test) ** 2

    Yref = _highband_fb @ Pref
    Ytest = _highband_fb @ Ptest

    # per-band -30 dB noise floor relative to the reference band peak over time
    nf = np.max(Yref, axis=1) * (10 ** (-30 / 10))
    Yref = np.maximum(Yref, nf.reshape(-1, 1)) ** 0.25
    Ytest = np.maximum(Ytest, nf.reshape(-1, 1)) ** 0.25

    delta = np.abs(Yref - Ytest)
    delta = np.linalg.norm(delta / delta.shape[-1], ord=2, axis=1)
    return 1000 * delta[-HIGHBAND_NUM_BANDS:]


def highband_compare(x_ref, x_test, fs=48000, threshold=0.0):
    """ Highband (bandwidth-extension) pass/fail against the lowpass anchor.

    A clip passes when, in every evaluated highband, the reference-to-test
    distortion stays at least `threshold` below the reference-to-lowpass anchor
    (the reference with the highband zeroed, i.e. "no extension"):

        pass  <=>  min_band( dist_ref_lp - dist_ref_test ) >= threshold

    Args:
        x_ref (np.ndarray): reference (fullband) signal scaled to [-1, 1]
        x_test (np.ndarray): decoded/extended test signal scaled to [-1, 1]
        fs (int): sampling rate of both signals (must be 48000)
        threshold (float): pass margin tau (default 0)

    Returns:
        (bool, np.ndarray | None): (passed, margins), margins = dist_ref_lp -
        dist_ref_test per evaluated highband (positive => beats the anchor).
        margins is None for a silent reference.
    """
    if fs != 48000:
        raise ValueError(f"highband evaluation requires 48 kHz signals, got fs={fs}")

    n = min(len(x_ref), len(x_test))
    x_ref = x_ref[:n].astype(np.float64)
    x_test = x_test[:n].astype(np.float64)

    m = np.max(np.abs(x_ref))
    if m == 0:
        return False, None
    x_ref = x_ref / m
    x_test = x_test / m

    _, _, X_ref = scipy.signal.stft(x_ref, nperseg=HIGHBAND_NPERSEG)
    _, _, X_test = scipy.signal.stft(x_test, nperseg=HIGHBAND_NPERSEG)

    # lowpass anchor: reference with the highband removed (no extension)
    X_lp = X_ref.copy()
    X_lp[HIGHBAND_LP_BIN:, :] = 0

    dist_ref_test = _highband_band_distortion(X_ref, X_test)
    dist_ref_lp = _highband_band_distortion(X_ref, X_lp)

    margins = dist_ref_lp - dist_ref_test
    passed = bool(np.all(margins >= threshold))
    return passed, margins

if __name__ == "__main__":
    import argparse
    from scipy.io import wavfile

    parser = argparse.ArgumentParser()
    parser.add_argument('ref', type=str, help='reference file (.wav or .s16)')
    parser.add_argument('deg', type=str, help='degraded file (.wav or .s16)')
    parser.add_argument('--fs-ref', type=int, default=16000, help='sampling rate of the reference for raw .s16 input (default: 16000; .wav uses its header)')
    parser.add_argument('--fs-deg', type=int, default=16000, help='sampling rate of the degraded/test signal for raw .s16 input (default: 16000; .wav uses its header)')
    parser.add_argument('--apply-vad', action='store_true')
    args = parser.parse_args()

    def load_signal(path, fs_default):
        if path.endswith(".s16"):
            return np.fromfile(path, dtype=np.int16), fs_default
        elif path.endswith(".wav"):
            fs, sig = wavfile.read(path)
            return sig, fs
        else:
            parser.print_help()
            sys.exit(1)

    x, fs1 = load_signal(args.ref, args.fs_ref)
    y, fs2 = load_signal(args.deg, args.fs_deg)

    for fs in (fs1, fs2):
        if fs % 16000 != 0:
            raise ValueError(f'error: sampling frequency {fs} is not a multiple of 16000')

    x = x.astype(np.float32) / 2**15
    y = y.astype(np.float32) / 2**15

    err = compare(x, y, fs_x=fs1, fs_y=fs2, apply_vad=args.apply_vad)

    print(f"MOC: {err}")
