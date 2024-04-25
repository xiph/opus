"""
/* Copyright (c) 2024 Amazon
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

import os

import numpy as np
import torch

import scipy
import scipy.signal
from scipy.io import wavfile

from utils.spec import log_spectrum, instafreq, create_filter_bank

def bwe_feature_factory(
    spec_num_bands=32,
    max_instafreq_bin=40
):
    """ features for bwe; we work with a fixed window size of 320 and a hop size of 160 """

    w = scipy.signal.windows.cosine(320)
    fb = create_filter_bank(spec_num_bands, 320, scale='erb', round_center_bins=True, normalize=True)

    def create_features(x, history=None):
        if history is None:
            history = np.zeros(320, dtype=np.float32)
        lmspec = log_spectrum(np.concatenate((history[-160:], x), dtype=x.dtype), frame_size=320, window=w, fb=fb)
        freqs = instafreq(np.concatenate((history[-320:], x), dtype=x.dtype), frame_size=320, max_bin=max_instafreq_bin, window=w)

        features = np.concatenate((lmspec, freqs), axis=-1, dtype=np.float32)

        return features

    return create_features


def load_inference_data(path,
                        spec_num_bands=32,
                        max_instafreq_bin=40,
                        **kwargs):

    print(f"[load_inference_data]: ignoring keyword arguments {kwargs.keys()}...")

    if path.endswith(".wav"):
        signal = wavfile.read(path)[1].astype(np.float32) / (2 ** 15)
    else:
        signal  = np.fromfile(path, dtype=np.int16).astype(np.float32) / (2 ** 15)

    num_frames = len(signal) // 160
    signal = signal[:num_frames*160]
    history = np.zeros(320, dtype=np.float32)

    create_features = bwe_feature_factory(spec_num_bands=spec_num_bands, max_instafreq_bin=max_instafreq_bin)

    features = create_features(signal, history)

    return torch.from_numpy(signal), torch.from_numpy(features)