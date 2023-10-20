import numpy as np
import scipy.signal

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


def compare(x, y):
    """ Modified version of opus_compare for 16 kHz mono signals

    Args:
        x (np.ndarray): reference input signal scaled to [-1, 1]
        y (np.ndarray): test signal scaled to [-1, 1]

    Returns:
        float: perceptually weighted error
    """
    # filter bank: bark scale with minimum-2-bin bands and cutoff at 7.5 kHz
    band_limits = [0, 2, 4, 6, 7, 9, 11, 13, 15, 18, 22, 26, 31, 36, 43, 51, 60, 75]
    num_bands = len(band_limits) - 1
    fb = rect_fb(band_limits, num_bins=81)

    # trim samples to same size
    num_samples = min(len(x), len(y))
    x = x[:num_samples] * 2**15
    y = y[:num_samples] * 2**15

    psd_x = power_spectrum(x) + 100000
    psd_y = power_spectrum(y) + 100000

    num_frames = psd_x.shape[0]

    # average band energies
    be_x = (psd_x @ fb.T) / np.sum(fb, axis=1)

    # frequecy masking
    f_mask = frequency_mask(num_bands, 0.1, 0.03)
    mask_x = be_x @ f_mask.T

    # temporal masking
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
    im = re - np.log(re) - 1
    Eb = ((im @ fb.T) / np.sum(fb, axis=1))
    Ef = np.mean(Eb ** 2, axis=1)
    err = np.mean(Ef ** 4, axis=0) ** (1/16)

    return float(err)