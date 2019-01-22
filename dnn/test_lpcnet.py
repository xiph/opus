#!/usr/bin/python3
'''Copyright (c) 2018 Mozilla

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

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

model, enc, dec = lpcnet.new_lpcnet_model(use_gpu=False)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#model.summary()

feature_file = sys.argv[1]
out_file = sys.argv[2]
frame_size = model.frame_size
nb_features = 55
nb_used_features = model.nb_used_features

features = np.fromfile(feature_file, dtype='float32')
features = np.resize(features, (-1, nb_features))
nb_frames = 1
feature_chunk_size = features.shape[0]
pcm_chunk_size = frame_size*feature_chunk_size

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features[:,:,18:36] = 0
periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')



model.load_weights('lpcnet20h_384_10_G16_80.h5')

order = 16

pcm = np.zeros((nb_frames*pcm_chunk_size, ))
fexc = np.zeros((1, 1, 3), dtype='int16')+128
state1 = np.zeros((1, model.rnn_units1), dtype='float32')
state2 = np.zeros((1, model.rnn_units2), dtype='float32')

mem = 0
coef = 0.85

fout = open(out_file, 'wb')

skip = order + 1
for c in range(0, nb_frames):
    cfeat = enc.predict([features[c:c+1, :, :nb_used_features], periods[c:c+1, :, :]])
    for fr in range(0, feature_chunk_size):
        f = c*feature_chunk_size + fr
        a = features[c, fr, nb_features-order:]
        for i in range(skip, frame_size):
            pred = -sum(a*pcm[f*frame_size + i - 1:f*frame_size + i - order-1:-1])
            fexc[0, 0, 1] = lin2ulaw(pred)

            p, state1, state2 = dec.predict([fexc, cfeat[:, fr:fr+1, :], state1, state2])
            #Lower the temperature for voiced frames to reduce noisiness
            p *= np.power(p, np.maximum(0, 1.5*features[c, fr, 37] - .5))
            p = p/(1e-18 + np.sum(p))
            #Cut off the tail of the remaining distribution
            p = np.maximum(p-0.002, 0).astype('float64')
            p = p/(1e-8 + np.sum(p))

            fexc[0, 0, 2] = np.argmax(np.random.multinomial(1, p[0,0,:], 1))
            pcm[f*frame_size + i] = pred + ulaw2lin(fexc[0, 0, 2])
            fexc[0, 0, 0] = lin2ulaw(pcm[f*frame_size + i])
            mem = coef*mem + pcm[f*frame_size + i]
            #print(mem)
            np.array([np.round(mem)], dtype='int16').tofile(fout)
        skip = 0


