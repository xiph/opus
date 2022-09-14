#!/usr/bin/python3
'''Copyright (c) 2021-2022 Amazon
   Copyright (c) 2018-2019 Mozilla

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
#from plc_loader import PLCLoader

parser = argparse.ArgumentParser(description='Train a PLC model')

parser.add_argument('bits', metavar='<bits file>', help='binary features file (int16)')
parser.add_argument('output', metavar='<output>', help='output features')
parser.add_argument('--model', metavar='<model>', default='rdovae', help='PLC model python definition (without .py)')
group1 = parser.add_mutually_exclusive_group()
group1.add_argument('--weights', metavar='<input weights>', help='model weights')
parser.add_argument('--cond-size', metavar='<units>', default=1024, type=int, help='number of units in conditioning network (default 1024)')
parser.add_argument('--batch-size', metavar='<batch size>', default=1, type=int, help='batch size to use (default 128)')
parser.add_argument('--seq-length', metavar='<sequence length>', default=1000, type=int, help='sequence length to use (default 1000)')


args = parser.parse_args()

import importlib
rdovae = importlib.import_module(args.model)

import sys
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
import h5py

import tensorflow as tf

# Try reducing batch_size if you run out of memory on your GPU
batch_size = args.batch_size

model, encoder, decoder = rdovae.new_rdovae_model(nb_used_features=20, nb_bits=80, batch_size=batch_size, cond_size=args.cond_size)
model.load_weights(args.weights)

lpc_order = 16

bits_file = args.bits
sequence_size = args.seq_length

# u for unquantised, load 16 bit PCM samples and convert to mu-law


bits = np.memmap(bits_file + "-bits.s16", dtype='int16', mode='r')
nb_sequences = len(bits)//(40*sequence_size)//batch_size*batch_size
bits = bits[:nb_sequences*sequence_size*40]

bits = np.reshape(bits, (nb_sequences, sequence_size//2, 20*4))
bits = bits[:,1::2,:]
print(bits.shape)

quant = np.memmap(bits_file + "-quant.f32", dtype='float32', mode='r')
state = np.memmap(bits_file + "-state.f32", dtype='float32', mode='r')

quant = np.reshape(quant, (nb_sequences, sequence_size//2, 6*20*4))
quant = quant[:,1::2,:]

state = np.reshape(state, (nb_sequences, sequence_size//2, 24))
state = state[:,-1,:]

print("shapes are:")
print(bits.shape)
print(quant.shape)
print(state.shape)

features = decoder.predict([bits, quant, state], batch_size=batch_size)

features.astype('float32').tofile(args.output)
