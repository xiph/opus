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

from torch.utils.data import Dataset
import numpy as np

from utils.bwe_features import bwe_feature_factory


class SimpleBWESet(Dataset):
    FRAME_SIZE_16K = 160
    def __init__(self,
                 path,
                 frames_per_sample=100,
                 spec_num_bands=32,
                 max_instafreq_bin=40,
                 upsampling_delay48=13,
                 ):

        self.frames_per_sample = frames_per_sample
        self.upsampling_delay48 = upsampling_delay48

        self.signal_16k = np.fromfile(os.path.join(path, 'signal_16kHz.s16'), dtype=np.int16)
        self.signal_48k = np.fromfile(os.path.join(path, 'signal_48kHz.s16'), dtype=np.int16)

        num_frames = min(len(self.signal_16k) // self.FRAME_SIZE_16K,
                         len(self.signal_48k) // (3 * self.FRAME_SIZE_16K))

        self.create_features = bwe_feature_factory(spec_num_bands=spec_num_bands, max_instafreq_bin=max_instafreq_bin)

        self.frame_offset = 6

        self.len = (num_frames - self.frame_offset) // frames_per_sample

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        frame_start = self.frames_per_sample * index + self.frame_offset
        frame_stop  = frame_start + self.frames_per_sample

        signal_start16 = frame_start * self.FRAME_SIZE_16K
        signal_stop16  = frame_stop  * self.FRAME_SIZE_16K

        x_16 = self.signal_16k[signal_start16 : signal_stop16].astype(np.float32) / 2**15
        history_16 = self.signal_16k[signal_start16 - 320 : signal_start16].astype(np.float32) / 2**15

        # dithering
        x_16 += (np.random.rand(len(x_16)) - 0.5) / 2**15
        history_16 += (np.random.rand(len(history_16)) - 0.5) / 2**15

        x_48 = self.signal_48k[3 * signal_start16 - self.upsampling_delay48
                               : 3 * signal_stop16 - self.upsampling_delay48].astype(np.float32) / 2**15

        features = self.create_features(
              x_16,
              history_16
        )

        return {
            'features'    : features,
            'x_16'        : x_16.astype(np.float32),
            'x_48'        : x_48.astype(np.float32),
            }
