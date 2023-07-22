import numpy as np
from scipy.io import wavfile
import amfm_decompy.pYAAPT as pYAAPT
import amfm_decompy.basic_tools as basic

def get_voicing_info(x, sr=16000):

    signal = basic.SignalObj(x, sr)
    pitch = pYAAPT.yaapt(signal, **{'frame_length' : 20.0, 'tda_frame_length' : 20.0})

    pitch_values = pitch.samp_values
    voiced_flags = pitch.vuv.astype('float')

    return pitch_values, voiced_flags

def compute_pitch_error(ref_path, test_path, fs=16000):
    fs_orig, x_orig = wavfile.read(ref_path)
    fs_test, x_test = wavfile.read(test_path)

    min_length = min(len(x_orig), len(x_test))
    x_orig = x_orig[:min_length]
    x_test = x_test[:min_length]

    assert fs_orig == fs_test == fs

    pitch_contour_orig, voicing_orig = get_voicing_info(x_orig.astype(np.float32))
    pitch_contour_test, voicing_test = get_voicing_info(x_test.astype(np.float32))

    return {
        'pitch_error' : np.mean(np.abs(pitch_contour_orig - pitch_contour_test)).item(),
        'voicing_error' : np.sum(np.abs(voicing_orig - voicing_test)).item() / len(voicing_orig)
        }