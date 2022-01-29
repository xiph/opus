#!/usr/bin/python3
'''Copyright (c) 2021-2022 Amazon

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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from tensorflow.keras.utils import Sequence

class PLCLoader(Sequence):
    def __init__(self, features, batch_size):
        self.batch_size = batch_size
        self.nb_batches = features.shape[0]//self.batch_size
        self.features = features[:self.nb_batches*self.batch_size, :]
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(self.nb_batches*self.batch_size)
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        features = self.features[self.indices[index*self.batch_size:(index+1)*self.batch_size], :, :]
        lost = (np.random.rand(features.shape[0], features.shape[1]) > .2).astype('float')
        lost = np.reshape(lost, (features.shape[0], features.shape[1], 1))
        lost_mask = np.tile(lost, (1,1,features.shape[2]))

        out_features = np.concatenate([features, 1.-lost], axis=-1)
        inputs = [features*lost_mask, lost]
        outputs = [out_features]
        return (inputs, outputs)

    def __len__(self):
        return self.nb_batches
