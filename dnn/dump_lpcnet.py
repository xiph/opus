#!/usr/bin/python3
'''Copyright (c) 2017-2018 Mozilla

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
from keras.layers import Layer, GRU, CuDNNGRU, Dense, Conv1D, Embedding
from ulaw import ulaw2lin, lin2ulaw
from mdense import MDense
import keras.backend as K
import h5py
import re

max_rnn_neurons = 1
max_conv_inputs = 1
max_mdense_tmp = 1

def printVector(f, vector, name, dtype='float'):
    v = np.reshape(vector, (-1));
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const {} {}[{}] = {{\n   '.format(dtype, name, len(v)))
    for i in range(0, len(v)):
        f.write('{}'.format(v[i]))
        if (i!=len(v)-1):
            f.write(',')
        else:
            break;
        if (i%8==7):
            f.write("\n   ")
        else:
            f.write(" ")
    #print(v, file=f)
    f.write('\n};\n\n')
    return;

def printSparseVector(f, A, name):
    N = A.shape[0]
    W = np.zeros((0,))
    diag = np.concatenate([np.diag(A[:,:N]), np.diag(A[:,N:2*N]), np.diag(A[:,2*N:])])
    A[:,:N] = A[:,:N] - np.diag(np.diag(A[:,:N]))
    A[:,N:2*N] = A[:,N:2*N] - np.diag(np.diag(A[:,N:2*N]))
    A[:,2*N:] = A[:,2*N:] - np.diag(np.diag(A[:,2*N:]))
    printVector(f, diag, name + '_diag')
    idx = np.zeros((0,), dtype='int')
    for i in range(3*N//16):
        pos = idx.shape[0]
        idx = np.append(idx, -1)
        nb_nonzero = 0
        for j in range(N):
            if np.sum(np.abs(A[j, i*16:(i+1)*16])) > 1e-10:
                nb_nonzero = nb_nonzero + 1
                idx = np.append(idx, j)
                W = np.concatenate([W, A[j, i*16:(i+1)*16]])
        idx[pos] = nb_nonzero
    printVector(f, W, name)
    #idx = np.tile(np.concatenate([np.array([N]), np.arange(N)]), 3*N//16)
    printVector(f, idx, name + '_idx', dtype='int')
    return;

def dump_layer_ignore(self, f, hf):
    print("ignoring layer " + self.name + " of type " + self.__class__.__name__)
    return False
Layer.dump_layer = dump_layer_ignore

def dump_sparse_gru(self, f, hf):
    global max_rnn_neurons
    name = 'sparse_' + self.name
    print("printing layer " + name + " of type sparse " + self.__class__.__name__)
    weights = self.get_weights()
    printSparseVector(f, weights[1], name + '_recurrent_weights')
    printVector(f, weights[-1], name + '_bias')
    if hasattr(self, 'activation'):
        activation = self.activation.__name__.upper()
    else:
        activation = 'TANH'
    if hasattr(self, 'reset_after') and not self.reset_after:
        reset_after = 0
    else:
        reset_after = 1
    neurons = weights[0].shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const SparseGRULayer {} = {{\n   {}_bias,\n   {}_recurrent_weights_diag,\n   {}_recurrent_weights,\n   {}_recurrent_weights_idx,\n   {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, name, weights[0].shape[1]//3, activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), weights[0].shape[1]//3))
    hf.write('extern const SparseGRULayer {};\n\n'.format(name));
    return True

def dump_gru_layer(self, f, hf):
    global max_rnn_neurons
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, weights[0], name + '_weights')
    printVector(f, weights[1], name + '_recurrent_weights')
    printVector(f, weights[-1], name + '_bias')
    if hasattr(self, 'activation'):
        activation = self.activation.__name__.upper()
    else:
        activation = 'TANH'
    if hasattr(self, 'reset_after') and not self.reset_after:
        reset_after = 0
    else:
        reset_after = 1
    neurons = weights[0].shape[1]//3
    max_rnn_neurons = max(max_rnn_neurons, neurons)
    f.write('const GRULayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_recurrent_weights,\n   {}, {}, ACTIVATION_{}, {}\n}};\n\n'
            .format(name, name, name, name, weights[0].shape[0], weights[0].shape[1]//3, activation, reset_after))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[1]//3))
    hf.write('#define {}_STATE_SIZE {}\n'.format(name.upper(), weights[0].shape[1]//3))
    hf.write('extern const GRULayer {};\n\n'.format(name));
    return True
CuDNNGRU.dump_layer = dump_gru_layer
GRU.dump_layer = dump_gru_layer

def dump_dense_layer_impl(name, weights, bias, activation, f, hf):
    printVector(f, weights, name + '_weights')
    printVector(f, bias, name + '_bias')
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weights.shape[0], weights.shape[1], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));

def dump_dense_layer(self, f, hf):
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    activation = self.activation.__name__.upper()
    dump_dense_layer_impl(name, weights[0], weights[1], activation, f, hf)
    return False

Dense.dump_layer = dump_dense_layer

def dump_mdense_layer(self, f, hf):
    global max_mdense_tmp
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, np.transpose(weights[0], (1, 2, 0)), name + '_weights')
    printVector(f, np.transpose(weights[1], (1, 0)), name + '_bias')
    printVector(f, np.transpose(weights[2], (1, 0)), name + '_factor')
    activation = self.activation.__name__.upper()
    max_mdense_tmp = max(max_mdense_tmp, weights[0].shape[0]*weights[0].shape[2])
    f.write('const MDenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}_factor,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, name, weights[0].shape[1], weights[0].shape[0], weights[0].shape[2], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[0]))
    hf.write('extern const MDenseLayer {};\n\n'.format(name));
    return False
MDense.dump_layer = dump_mdense_layer

def dump_conv1d_layer(self, f, hf):
    global max_conv_inputs
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, weights[0], name + '_weights')
    printVector(f, weights[-1], name + '_bias')
    activation = self.activation.__name__.upper()
    max_conv_inputs = max(max_conv_inputs, weights[0].shape[1]*weights[0].shape[0])
    f.write('const Conv1DLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weights[0].shape[1], weights[0].shape[0], weights[0].shape[2], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[2]))
    hf.write('#define {}_STATE_SIZE ({}*{})\n'.format(name.upper(), weights[0].shape[1], (weights[0].shape[0]-1)))
    hf.write('#define {}_DELAY {}\n'.format(name.upper(), (weights[0].shape[0]-1)//2))
    hf.write('extern const Conv1DLayer {};\n\n'.format(name));
    return True
Conv1D.dump_layer = dump_conv1d_layer


def dump_embedding_layer_impl(name, weights, f, hf):
    printVector(f, weights, name + '_weights')
    f.write('const EmbeddingLayer {} = {{\n   {}_weights,\n   {}, {}\n}};\n\n'
            .format(name, name, weights.shape[0], weights.shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights.shape[1]))
    hf.write('extern const EmbeddingLayer {};\n\n'.format(name));

def dump_embedding_layer(self, f, hf):
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()[0]
    dump_embedding_layer_impl(name, weights, f, hf)
    return False
Embedding.dump_layer = dump_embedding_layer


model, _, _ = lpcnet.new_lpcnet_model(rnn_units1=384, use_gpu=False)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
#model.summary()

model.load_weights(sys.argv[1])

if len(sys.argv) > 2:
    cfile = sys.argv[2];
    hfile = sys.argv[3];
else:
    cfile = 'nnet_data.c'
    hfile = 'nnet_data.h'


f = open(cfile, 'w')
hf = open(hfile, 'w')


f.write('/*This file is automatically generated from a Keras model*/\n\n')
f.write('#ifdef HAVE_CONFIG_H\n#include "config.h"\n#endif\n\n#include "nnet.h"\n#include "{}"\n\n'.format(hfile))

hf.write('/*This file is automatically generated from a Keras model*/\n\n')
hf.write('#ifndef RNN_DATA_H\n#define RNN_DATA_H\n\n#include "nnet.h"\n\n')

embed_size = lpcnet.embed_size

E = model.get_layer('embed_sig').get_weights()[0]
W = model.get_layer('gru_a').get_weights()[0][:embed_size,:]
dump_embedding_layer_impl('gru_a_embed_sig', np.dot(E, W), f, hf)
W = model.get_layer('gru_a').get_weights()[0][embed_size:2*embed_size,:]
dump_embedding_layer_impl('gru_a_embed_pred', np.dot(E, W), f, hf)
W = model.get_layer('gru_a').get_weights()[0][2*embed_size:3*embed_size,:]
dump_embedding_layer_impl('gru_a_embed_exc', np.dot(E, W), f, hf)
W = model.get_layer('gru_a').get_weights()[0][3*embed_size:,:]
#FIXME: dump only half the biases
b = model.get_layer('gru_a').get_weights()[2]
dump_dense_layer_impl('gru_a_dense_feature', W, b, 'LINEAR', f, hf)

layer_list = []
for i, layer in enumerate(model.layers):
    if layer.dump_layer(f, hf):
        layer_list.append(layer.name)

dump_sparse_gru(model.get_layer('gru_a'), f, hf)

hf.write('#define MAX_RNN_NEURONS {}\n\n'.format(max_rnn_neurons))
hf.write('#define MAX_CONV_INPUTS {}\n\n'.format(max_conv_inputs))
hf.write('#define MAX_MDENSE_TMP {}\n\n'.format(max_mdense_tmp))


hf.write('typedef struct {\n')
for i, name in enumerate(layer_list):
    hf.write('  float {}_state[{}_STATE_SIZE];\n'.format(name, name.upper())) 
hf.write('} NNetState;\n')

hf.write('\n\n#endif\n')

f.close()
hf.close()
