import pesq
import librosa

def compute_PESQ(ref, test, fs=16000):

    if not ref.endswith('.wav') or not test.endswith('.wav'):
        raise ValueError('error: expecting .wav as file extension')

    ref_item, _ = librosa.load(ref, sr=fs)
    test_item, _ = librosa.load(test, sr=fs)

    score = pesq.pesq(fs, ref_item, test_item)

    return score