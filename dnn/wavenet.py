#!/usr/bin/python3

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNGRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Add, Multiply, Bidirectional, MaxPooling1D, Activation
from keras import backend as K
from keras.initializers import Initializer
from keras.initializers import VarianceScaling
from mdense import MDense
import numpy as np
import h5py
import sys
from causalconv import CausalConv
from gatedconv import GatedConv

units=128
pcm_bits = 8
pcm_levels = 2**pcm_bits
nb_used_features = 38

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

def new_wavenet_model(fftnet=False):
    pcm = Input(shape=(None, 1))
    pitch = Input(shape=(None, 1))
    feat = Input(shape=(None, nb_used_features))
    dec_feat = Input(shape=(None, 32))

    fconv1 = Conv1D(128, 3, padding='same', activation='tanh')
    fconv2 = Conv1D(32, 3, padding='same', activation='tanh')

    cfeat = fconv2(fconv1(feat))

    rep = Lambda(lambda x: K.repeat_elements(x, 160, 1))

    activation='tanh'
    rfeat = rep(cfeat)
    #tmp = Concatenate()([pcm, rfeat])
    embed = Embedding(256, units, embeddings_initializer=PCMInit())
    tmp = Reshape((-1, units))(embed(pcm))
    init = VarianceScaling(scale=1.5,mode='fan_avg',distribution='uniform')
    for k in range(10):
        res = tmp
        dilation = 9-k if fftnet else k
        tmp = Concatenate()([tmp, rfeat])
        c = GatedConv(units, 2, dilation_rate=2**dilation, activation='tanh', kernel_initializer=init)
        tmp = Dense(units, activation='relu')(c(tmp))
        
        '''tmp = Concatenate()([tmp, rfeat])
        c1 = CausalConv(units, 2, dilation_rate=2**dilation, activation='tanh')
        c2 = CausalConv(units, 2, dilation_rate=2**dilation, activation='sigmoid')
        tmp = Multiply()([c1(tmp), c2(tmp)])
        tmp = Dense(units, activation='relu')(tmp)'''
        
        if k != 0:
            tmp = Add()([tmp, res])

    md = MDense(pcm_levels, activation='softmax')
    ulaw_prob = md(tmp)
    
    model = Model([pcm, feat], ulaw_prob)
    return model
