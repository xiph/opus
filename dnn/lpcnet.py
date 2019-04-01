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

import math
from keras.models import Model
from keras.layers import Input, GRU, CuDNNGRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Add, Bidirectional, MaxPooling1D, Activation
from keras import backend as K
from keras.initializers import Initializer
from keras.callbacks import Callback
from mdense import MDense
import numpy as np
import h5py
import sys

frame_size = 160
pcm_bits = 8
embed_size = 128
pcm_levels = 2**pcm_bits

class Sparsify(Callback):
    def __init__(self, t_start, t_end, interval, density):
        super(Sparsify, self).__init__()
        self.batch = 0
        self.t_start = t_start
        self.t_end = t_end
        self.interval = interval
        self.final_density = density

    def on_batch_end(self, batch, logs=None):
        #print("batch number", self.batch)
        self.batch += 1
        if self.batch < self.t_start or ((self.batch-self.t_start) % self.interval != 0 and self.batch < self.t_end):
            #print("don't constrain");
            pass
        else:
            #print("constrain");
            layer = self.model.get_layer('gru_a')
            w = layer.get_weights()
            p = w[1]
            nb = p.shape[1]//p.shape[0]
            N = p.shape[0]
            #print("nb = ", nb, ", N = ", N);
            #print(p.shape)
            #print ("density = ", density)
            for k in range(nb):
                density = self.final_density[k]
                if self.batch < self.t_end:
                    r = 1 - (self.batch-self.t_start)/(self.t_end - self.t_start)
                    density = 1 - (1-self.final_density[k])*(1 - r*r*r)
                A = p[:, k*N:(k+1)*N]
                A = A - np.diag(np.diag(A))
                A = np.transpose(A, (1, 0))
                L=np.reshape(A, (N, N//16, 16))
                S=np.sum(L*L, axis=-1)
                SS=np.sort(np.reshape(S, (-1,)))
                thresh = SS[round(N*N//16*(1-density))]
                mask = (S>=thresh).astype('float32');
                mask = np.repeat(mask, 16, axis=1)
                mask = np.minimum(1, mask + np.diag(np.ones((N,))))
                mask = np.transpose(mask, (1, 0))
                p[:, k*N:(k+1)*N] = p[:, k*N:(k+1)*N]*mask
                #print(thresh, np.mean(mask))
            w[1] = p
            layer.set_weights(w)
            

class PCMInit(Initializer):
    def __init__(self, gain=.1, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=None):
        num_rows = 1
        for dim in shape[:-1]:
            num_rows *= dim
        num_cols = shape[-1]
        flat_shape = (num_rows, num_cols)
        if self.seed is not None:
            np.random.seed(self.seed)
        a = np.random.uniform(-1.7321, 1.7321, flat_shape)
        #a[:,0] = math.sqrt(12)*np.arange(-.5*num_rows+.5,.5*num_rows-.4)/num_rows
        #a[:,1] = .5*a[:,0]*a[:,0]*a[:,0]
        a = a + np.reshape(math.sqrt(12)*np.arange(-.5*num_rows+.5,.5*num_rows-.4)/num_rows, (num_rows, 1))
        return self.gain * a

    def get_config(self):
        return {
            'gain': self.gain,
            'seed': self.seed
        }

def new_lpcnet_model(rnn_units1=384, rnn_units2=16, nb_used_features = 38, training=False, use_gpu=True, adaptation=False):
    pcm = Input(shape=(None, 3))
    feat = Input(shape=(None, nb_used_features))
    pitch = Input(shape=(None, 1))
    dec_feat = Input(shape=(None, 128))
    dec_state1 = Input(shape=(rnn_units1,))
    dec_state2 = Input(shape=(rnn_units2,))

    padding = 'valid' if training else 'same'
    fconv1 = Conv1D(128, 3, padding=padding, activation='tanh', name='feature_conv1')
    fconv2 = Conv1D(128, 3, padding=padding, activation='tanh', name='feature_conv2')

    embed = Embedding(256, embed_size, embeddings_initializer=PCMInit(), name='embed_sig')
    cpcm = Reshape((-1, embed_size*3))(embed(pcm))

    pembed = Embedding(256, 64, name='embed_pitch')
    cat_feat = Concatenate()([feat, Reshape((-1, 64))(pembed(pitch))])
    
    cfeat = fconv2(fconv1(cat_feat))

    fdense1 = Dense(128, activation='tanh', name='feature_dense1')
    fdense2 = Dense(128, activation='tanh', name='feature_dense2')

    cfeat = fdense2(fdense1(cfeat))
    
    rep = Lambda(lambda x: K.repeat_elements(x, frame_size, 1))

    if use_gpu:
        rnn = CuDNNGRU(rnn_units1, return_sequences=True, return_state=True, name='gru_a')
        rnn2 = CuDNNGRU(rnn_units2, return_sequences=True, return_state=True, name='gru_b')
    else:
        rnn = GRU(rnn_units1, return_sequences=True, return_state=True, recurrent_activation="sigmoid", reset_after='true', name='gru_a')
        rnn2 = GRU(rnn_units2, return_sequences=True, return_state=True, recurrent_activation="sigmoid", reset_after='true', name='gru_b')

    rnn_in = Concatenate()([cpcm, rep(cfeat)])
    md = MDense(pcm_levels, activation='softmax', name='dual_fc')
    gru_out1, _ = rnn(rnn_in)
    gru_out2, _ = rnn2(Concatenate()([gru_out1, rep(cfeat)]))
    ulaw_prob = md(gru_out2)
    
    if adaptation:
        rnn.trainable=False
        rnn2.trainable=False
        md.trainable=False
        embed.Trainable=False
    
    model = Model([pcm, feat, pitch], ulaw_prob)
    model.rnn_units1 = rnn_units1
    model.rnn_units2 = rnn_units2
    model.nb_used_features = nb_used_features
    model.frame_size = frame_size

    encoder = Model([feat, pitch], cfeat)
    
    dec_rnn_in = Concatenate()([cpcm, dec_feat])
    dec_gru_out1, state1 = rnn(dec_rnn_in, initial_state=dec_state1)
    dec_gru_out2, state2 = rnn2(Concatenate()([dec_gru_out1, dec_feat]), initial_state=dec_state2)
    dec_ulaw_prob = md(dec_gru_out2)

    decoder = Model([pcm, dec_feat, dec_state1, dec_state2], [dec_ulaw_prob, state1, state2])
    return model, encoder, decoder
