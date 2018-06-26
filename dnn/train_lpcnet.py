#!/usr/bin/python3

import lpcnet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.44
set_session(tf.Session(config=config))

nb_epochs = 40
batch_size = 64

model = lpcnet.new_wavernn_model()
model.compile(optimizer=Adam(0.0008), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

pcmfile = sys.argv[1]
feature_file = sys.argv[2]
nb_features = 54
nb_used_features = lpcnet.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = 160*feature_chunk_size

data = np.fromfile(pcmfile, dtype='int8')
nb_frames = len(data)//pcm_chunk_size

features = np.fromfile(feature_file, dtype='float32')

data = data[:nb_frames*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features]

in_data = np.concatenate([data[0:1], data[:-1]])/16.;

in_data = np.reshape(in_data, (nb_frames, pcm_chunk_size, 1))
out_data = np.reshape(data, (nb_frames, pcm_chunk_size, 1))
out_data = (out_data.astype('int16')+128).astype('uint8')
features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features = features[:, :, :nb_used_features]

checkpoint = ModelCheckpoint('lpcnet1c_{epoch:02d}.h5')

#model.load_weights('wavernn1c_01.h5')
model.compile(optimizer=Adam(0.002, amsgrad=True, decay=2e-4), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit([in_data, features], out_data, batch_size=batch_size, epochs=30, validation_split=0.2, callbacks=[checkpoint])
