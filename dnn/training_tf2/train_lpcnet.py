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

# Train an LPCNet model

import argparse

parser = argparse.ArgumentParser(description='Train an LPCNet model')

parser.add_argument('features', metavar='<features file>', help='binary features file (float32)')
parser.add_argument('data', metavar='<audio data file>', help='binary audio data file (uint8)')
parser.add_argument('output', metavar='<output>', help='trained model file (.h5)')
parser.add_argument('--model', metavar='<model>', default='lpcnet', help='LPCNet model python definition (without .py)')
parser.add_argument('--quantize', metavar='<input weights>', help='quantize model')
parser.add_argument('--density', metavar='<global density>', type=float, help='average density of the recurrent weights (default 0.1)')
parser.add_argument('--density-split', nargs=3, metavar=('<update>', '<reset>', '<state>'), type=float, help='density of each recurrent gate (default 0.05, 0.05, 0.2)')
parser.add_argument('--grub-density', metavar='<global GRU B density>', type=float, help='average density of the recurrent weights (default 1.0)')
parser.add_argument('--grub-density-split', nargs=3, metavar=('<update>', '<reset>', '<state>'), type=float, help='density of each GRU B input gate (default 1.0, 1.0, 1.0)')
parser.add_argument('--grua-size', metavar='<units>', default=384, type=int, help='number of units in GRU A (default 384)')
parser.add_argument('--grub-size', metavar='<units>', default=16, type=int, help='number of units in GRU B (default 16)')
parser.add_argument('--epochs', metavar='<epochs>', default=120, type=int, help='number of epochs to train for (default 120)')
parser.add_argument('--batch-size', metavar='<batch size>', default=128, type=int, help='batch size to use (default 128)')


args = parser.parse_args()

density = (0.05, 0.05, 0.2)
if args.density_split is not None:
    density = args.density_split
elif args.density is not None:
    density = [0.5*args.density, 0.5*args.density, 2.0*args.density];

grub_density = (1., 1., 1.)
if args.grub_density_split is not None:
    grub_density = args.grub_density_split
elif args.grub_density is not None:
    grub_density = [0.5*args.grub_density, 0.5*args.grub_density, 2.0*args.grub_density];

import importlib
lpcnet = importlib.import_module(args.model)

import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from ulaw import ulaw2lin, lin2ulaw
import tensorflow.keras.backend as K
import h5py

import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#  try:
#    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#  except RuntimeError as e:
#    print(e)

nb_epochs = args.epochs

# Try reducing batch_size if you run out of memory on your GPU
batch_size = args.batch_size

quantize = args.quantize is not None

if quantize:
    lr = 0.00003
    decay = 0
else:
    lr = 0.001
    decay = 2.5e-5

opt = Adam(lr, decay=decay, beta_2=0.99)
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

with strategy.scope():
    model, _, _ = lpcnet.new_lpcnet_model(rnn_units1=args.grua_size, rnn_units2=args.grub_size, training=True, quantize=quantize)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics='sparse_categorical_crossentropy')
    model.summary()

feature_file = args.features
pcm_file = args.data     # 16 bit unsigned short PCM samples
frame_size = model.frame_size
nb_features = 55
nb_used_features = model.nb_used_features
feature_chunk_size = 15
pcm_chunk_size = frame_size*feature_chunk_size

# u for unquantised, load 16 bit PCM samples and convert to mu-law

data = np.memmap(pcm_file, dtype='uint8', mode='r')
nb_frames = len(data)//(4*pcm_chunk_size)//batch_size*batch_size

features = np.memmap(feature_file, dtype='float32', mode='r')

# limit to discrete number of frames
data = data[:nb_frames*4*pcm_chunk_size]
features = features[:nb_frames*feature_chunk_size*nb_features].copy()

features = np.reshape(features, (nb_frames*feature_chunk_size, nb_features))

data = np.reshape(data, (nb_frames, pcm_chunk_size, 4))
in_data = data[:,:,:3]
out_exc = data[:,:,3:4]

print("ulaw std = ", np.std(out_exc))

features = np.reshape(features, (nb_frames, feature_chunk_size, nb_features))
features = features[:, :, :nb_used_features]
features[:,:,18:36] = 0

fpad1 = np.concatenate([features[0:1, 0:2, :], features[:-1, -2:, :]], axis=0)
fpad2 = np.concatenate([features[1:, :2, :], features[0:1, -2:, :]], axis=0)
features = np.concatenate([fpad1, features, fpad2], axis=1)


periods = (.1 + 50*features[:,:,36:37]+100).astype('int16')
#periods = np.minimum(periods, 255)

# dump models to disk as we go
checkpoint = ModelCheckpoint('{}_{}_{}.h5'.format(args.output, args.grua_size, '{epoch:02d}'))

if quantize:
    #Adapting from an existing model
    model.load_weights(args.quantize)
    sparsify = lpcnet.Sparsify(0, 0, 1, density)
    grub_sparsify = lpcnet.SparsifyGRUB(0, 0, 1, args.grua_size, grub_density)
else:
    #Training from scratch
    sparsify = lpcnet.Sparsify(2000, 40000, 400, density)
    grub_sparsify = lpcnet.SparsifyGRUB(2000, 40000, 400, args.grua_size, grub_density)

model.save_weights('{}_{}_initial.h5'.format(args.output, args.grua_size))
model.fit([in_data, features, periods], out_exc, batch_size=batch_size, epochs=nb_epochs, validation_split=0.0, callbacks=[checkpoint, sparsify, grub_sparsify])
