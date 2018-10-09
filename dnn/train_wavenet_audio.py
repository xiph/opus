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
config.gpu_options.per_process_gpu_memory_fraction = 0.44
set_session(tf.Session(config=config))

nb_epochs = 40
batch_size = 64

#model = wavenet.new_wavenet_model(fftnet=True)
model, _, _ = lpcnet.new_wavernn_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.summary()

exc_file = sys.argv[1]
feature_file = sys.argv[2]
pred_file = sys.argv[3]
pcm_file = sys.argv[4]
frame_size = 160
nb_features = 55
nb_used_features = lpcnet.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

udata = np.fromfile(pcm_file, dtype='int16')
data = lin2ulaw(udata)
nb_frames = len(data)//pcm_chunk_size

features = np.fromfile(feature_file, dtype='float32')

data = data[:nb_frames*pcm_chunk_size]
udata = udata[:nb_frames*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features]

in_data = np.concatenate([data[0:1], data[:-1]]);
noise = np.concatenate([np.zeros((len(data)*1//5)), np.random.randint(-3, 3, len(data)*1//5), np.random.randint(-2, 2, len(data)*1//5), np.random.randint(-1, 1, len(data)*2//5)])
in_data = in_data + noise
in_data = np.clip(in_data, 0, 255)

features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))

upred = np.fromfile(pred_file, dtype='int16')
upred = upred[:nb_frames*pcm_chunk_size]

pred_in = ulaw2lin(in_data)
for i in range(2, nb_frames*feature_chunk_size):
    upred[i*frame_size:(i+1)*frame_size] = 0
    #if i % 100000 == 0:
    #    print(i)
    for k in range(16):
        upred[i*frame_size:(i+1)*frame_size] = upred[i*frame_size:(i+1)*frame_size] - \
            pred_in[i*frame_size-k:(i+1)*frame_size-k]*features[i, nb_features-16+k]

pred = lin2ulaw(upred)
#pred = pred + np.random.randint(-1, 1, len(data))


in_data = np.reshape(in_data, (nb_frames, pcm_chunk_size, 1))
in_data = in_data.astype('uint8')
out_data = lin2ulaw(udata-upred)
in_exc = np.concatenate([out_data[0:1], out_data[:-1]]);

out_data = np.reshape(out_data, (nb_frames, pcm_chunk_size, 1))
out_data = out_data.astype('uint8')

in_exc = np.reshape(in_exc, (nb_frames, pcm_chunk_size, 1))
in_exc = in_exc.astype('uint8')


features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features = features[:, :, :nb_used_features]
pred = np.reshape(pred, (nb_frames, pcm_chunk_size, 1))
pred = pred.astype('uint8')

periods = (50*features[:,:,36:37]+100).astype('int16')

in_data = np.concatenate([in_data, pred], axis=-1)

#in_data = np.concatenate([in_data, in_pitch], axis=-1)

#with h5py.File('in_data.h5', 'w') as f:
# f.create_dataset('data', data=in_data[:50000, :, :])
# f.create_dataset('feat', data=features[:50000, :, :])

checkpoint = ModelCheckpoint('wavenet5b_{epoch:02d}.h5')

#model.load_weights('wavenet4f2_30.h5')
model.compile(optimizer=Adam(0.001, amsgrad=True, decay=2e-4), loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit([in_data, in_exc, features, periods], out_data, batch_size=batch_size, epochs=30, validation_split=0.2, callbacks=[checkpoint])
