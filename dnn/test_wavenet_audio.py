#!/usr/bin/python3

import wavenet
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

pcmfile = sys.argv[1]
feature_file = sys.argv[2]
frame_size = 160
nb_features = 55
nb_used_features = wavenet.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

data = np.fromfile(pcmfile, dtype='int16')
data = np.minimum(127, lin2ulaw(data/32768.))
nb_frames = len(data)//pcm_chunk_size

features = np.fromfile(feature_file, dtype='float32')

data = data[:nb_frames*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features]

in_data = np.concatenate([data[0:1], data[:-1]]);

features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))
pitch = 1.*data
pitch[:320] = 0
for i in range(2, nb_frames*feature_chunk_size):
    period = int(50*features[i,36]+100)
    period = period - 4
    pitch[i*frame_size:(i+1)*frame_size] = data[i*frame_size-period:(i+1)*frame_size-period]
in_pitch = np.reshape(pitch/16., (nb_frames, pcm_chunk_size, 1))

in_data = np.reshape(in_data, (nb_frames, pcm_chunk_size, 1))
in_data = (in_data.astype('int16')+128).astype('uint8')
out_data = np.reshape(data, (nb_frames, pcm_chunk_size, 1))
out_data = (out_data.astype('int16')+128).astype('uint8')
features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features = features[:, :, :]



in_data = np.reshape(in_data, (nb_frames*pcm_chunk_size, 1))
out_data = np.reshape(data, (nb_frames*pcm_chunk_size, 1))


model.load_weights('wavenet3h13_30.h5')

order = 16

pcm = 0.*out_data
fexc = np.zeros((1, 1, 2), dtype='float32')
iexc = np.zeros((1, 1, 1), dtype='int16')
state = np.zeros((1, lpcnet.rnn_units), dtype='float32')
for c in range(1, nb_frames):
    cfeat = enc.predict(features[c:c+1, :, :nb_used_features])
    for fr in range(1, feature_chunk_size):
        f = c*feature_chunk_size + fr
        a = features[c, fr, nb_features-order:]
        
        #print(a)
        gain = 1.;
        period = int(50*features[c, fr, 36]+100)
        period = period - 4
        for i in range(frame_size):
            fexc[0, 0, 0] = iexc + 128
            pred = -sum(a*pcm[f*frame_size + i - 1:f*frame_size + i - order-1:-1, 0])
            fexc[0, 0, 1] = np.minimum(127, lin2ulaw(pred/32768.)) + 128

            p, state = dec.predict([fexc, cfeat[:, fr:fr+1, :], state])
            p = np.maximum(p-0.001, 0)
            p = p/(1e-5 + np.sum(p))

            iexc[0, 0, 0] = np.argmax(np.random.multinomial(1, p[0,0,:], 1))-128
            pcm[f*frame_size + i, 0] = 32768*ulaw2lin(iexc[0, 0, 0]*1.0)
            print(iexc[0, 0, 0], out_data[f*frame_size + i, 0], pcm[f*frame_size + i, 0])


