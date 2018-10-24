#!/usr/bin/python3

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

nb_epochs = 40
batch_size = 64

#model = wavenet.new_wavenet_model(fftnet=True)
model, enc, dec = lpcnet.new_wavernn_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#model.summary()

feature_file = sys.argv[1]
frame_size = 160
nb_features = 55
nb_used_features = lpcnet.nb_used_features

features = np.fromfile(feature_file, dtype='float32')
features = np.resize(features, (-1, nb_features))
nb_frames = 1
feature_chunk_size = features.shape[0]
pcm_chunk_size = frame_size*feature_chunk_size

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features[:,:,18:36] = 0
periods = (50*features[:,:,36:37]+100).astype('int16')



model.load_weights('lpcnet9_384_10_G16_120.h5')

order = 16

pcm = np.zeros((nb_frames*pcm_chunk_size, ))
fexc = np.zeros((1, 1, 2), dtype='float32')
iexc = np.zeros((1, 1, 1), dtype='int16')
state1 = np.zeros((1, lpcnet.rnn_units1), dtype='float32')
state2 = np.zeros((1, lpcnet.rnn_units2), dtype='float32')

mem = 0
coef = 0.85

skip = order + 1
for c in range(0, nb_frames):
    cfeat = enc.predict([features[c:c+1, :, :nb_used_features], periods[c:c+1, :, :]])
    for fr in range(0, feature_chunk_size):
        f = c*feature_chunk_size + fr
        a = features[c, fr, nb_features-order:]
        for i in range(skip, frame_size):
            pred = -sum(a*pcm[f*frame_size + i - 1:f*frame_size + i - order-1:-1])
            fexc[0, 0, 1] = lin2ulaw(pred)

            p, state1, state2 = dec.predict([fexc, iexc, cfeat[:, fr:fr+1, :], state1, state2])
            #Lower the temperature for voiced frames to reduce noisiness
            p *= np.power(p, np.maximum(0, 1.5*features[c, fr, 37] - .5))
            p = p/(1e-18 + np.sum(p))
            #Cut off the tail of the remaining distribution
            p = np.maximum(p-0.002, 0).astype('float64')
            p = p/(1e-8 + np.sum(p))

            iexc[0, 0, 0] = np.argmax(np.random.multinomial(1, p[0,0,:], 1))
            pcm[f*frame_size + i] = pred + ulaw2lin(iexc[0, 0, 0])
            fexc[0, 0, 0] = lin2ulaw(pcm[f*frame_size + i])
            mem = coef*mem + pcm[f*frame_size + i]
            print(mem)
        skip = 0


