#!/usr/bin/python3

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py
from adadiff import Adadiff

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.28
#set_session(tf.Session(config=config))

nb_epochs = 40
batch_size = 64

model, enc, dec = lpcnet.new_wavernn_model()
model.compile(optimizer=Adadiff(), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

pcmfile = sys.argv[1]
feature_file = sys.argv[2]
frame_size = 160
nb_features = 54
nb_used_features = lpcnet.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

data = np.fromfile(pcmfile, dtype='int8')
nb_frames = len(data)//pcm_chunk_size

features = np.fromfile(feature_file, dtype='float32')

data = data[:nb_frames*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features]

in_data = np.concatenate([data[0:1], data[:-1]])/16.;

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))

in_data = np.reshape(in_data, (nb_frames*pcm_chunk_size, 1))
out_data = np.reshape(data, (nb_frames*pcm_chunk_size, 1))


model.load_weights('lpcnet1h_30.h5')

order = 16

pcm = 0.*out_data
for c in range(1, nb_frames):
    for fr in range(1, feature_chunk_size):
        f = c*feature_chunk_size + fr
        a = features[c, fr, nb_used_features+1:]
        #print(a)
        gain = 1;
        for i in range(frame_size):
            pcm[f*frame_size + i, 0] = gain*out_data[f*frame_size + i, 0] - sum(a*pcm[f*frame_size + i - 1:f*frame_size + i - order-1:-1, 0])
            print(pcm[f*frame_size + i, 0])

