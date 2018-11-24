#!/usr/bin/python3

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

def printVector(f, vector, name):
    v = np.reshape(vector, (-1));
    #print('static const float ', name, '[', len(v), '] = \n', file=f)
    f.write('static const float {}[{}] = {{\n   '.format(name, len(v)))
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

def dump_layer_ignore(self, f, hf):
    print("ignoring layer " + self.name + " of type " + self.__class__.__name__)
    return False
Layer.dump_layer = dump_layer_ignore

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

def dump_dense_layer(self, f, hf):
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, weights[0], name + '_weights')
    printVector(f, weights[-1], name + '_bias')
    activation = self.activation.__name__.upper()
    f.write('const DenseLayer {} = {{\n   {}_bias,\n   {}_weights,\n   {}, {}, ACTIVATION_{}\n}};\n\n'
            .format(name, name, name, weights[0].shape[0], weights[0].shape[1], activation))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[1]))
    hf.write('extern const DenseLayer {};\n\n'.format(name));
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
            .format(name, name, name, name, weights[0].shape[0], weights[0].shape[1], weights[0].shape[2], activation))
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
    hf.write('extern const Conv1DLayer {};\n\n'.format(name));
    return True
Conv1D.dump_layer = dump_conv1d_layer


def dump_embedding_layer(self, f, hf):
    name = self.name
    print("printing layer " + name + " of type " + self.__class__.__name__)
    weights = self.get_weights()
    printVector(f, weights[0], name + '_weights')
    f.write('const EmbeddingLayer {} = {{\n   {}_weights,\n   {}, {}\n}};\n\n'
            .format(name, name, weights[0].shape[0], weights[0].shape[1]))
    hf.write('#define {}_OUT_SIZE {}\n'.format(name.upper(), weights[0].shape[1]))
    hf.write('extern const EmbeddingLayer {};\n\n'.format(name));
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

layer_list = []
for i, layer in enumerate(model.layers):
    if layer.dump_layer(f, hf):
        layer_list.append(layer.name)

hf.write('#define MAX_RNN_NEURONS {}\n\n'.format(max_rnn_neurons))
hf.write('#define MAX_CONV_INPUTS {}\n\n'.format(max_conv_inputs))
hf.write('#define MAX_MDENSE_TMP {}\n\n'.format(max_mdense_tmp))


hf.write('struct RNNState {\n')
for i, name in enumerate(layer_list):
    hf.write('  float {}_state[{}_STATE_SIZE];\n'.format(name, name.upper())) 
hf.write('};\n')

hf.write('\n\n#endif\n')

f.close()
hf.close()
