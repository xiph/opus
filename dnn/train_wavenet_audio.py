#!/usr/bin/python3

import wavenet
import sys
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import keras.backend as K
import h5py

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.44
#set_session(tf.Session(config=config))

nb_epochs = 40
batch_size = 64

model = wavenet.new_wavenet_model(fftnet=True)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

pcmfile = sys.argv[1]
feature_file = sys.argv[2]
frame_size = 160
nb_features = 54
nb_used_features = wavenet.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

data = np.fromfile(pcmfile, dtype='int16')
data = np.minimum(127, lin2ulaw(data[80:]/32768.))
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
features = features[:, :, :nb_used_features]


#in_data = np.concatenate([in_data, in_pitch], axis=-1)

#with h5py.File('in_data.h5', 'w') as f:
# f.create_dataset('data', data=in_data[:50000, :, :])
# f.create_dataset('feat', data=features[:50000, :, :])

checkpoint = ModelCheckpoint('wavenet3c_{epoch:02d}.h5')

#model.load_weights('wavernn1c_01.h5')
model.compile(optimizer=Adam(0.001, amsgrad=True, decay=2e-4), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit([in_data, features], out_data, batch_size=batch_size, epochs=30, validation_split=0.2, callbacks=[checkpoint])
