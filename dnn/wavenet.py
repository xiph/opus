#!/usr/bin/python3

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNGRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Add, Multiply, Bidirectional, MaxPooling1D, Activation
from keras import backend as K
from mdense import MDense
import numpy as np
import h5py
import sys
from causalconv import CausalConv

units=128
pcm_bits = 8
pcm_levels = 2**pcm_bits
nb_used_features = 38


def new_wavenet_model():
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
    tmp = pcm
    for k in range(10):
        res = tmp
        tmp = Concatenate()([tmp, rfeat])
        c1 = CausalConv(units, 2, dilation_rate=2**k, activation='tanh')
        c2 = CausalConv(units, 2, dilation_rate=2**k, activation='sigmoid')
        tmp = Multiply()([c1(tmp), c2(tmp)])
        tmp = Dense(units, activation='relu')(tmp)
        if k != 0:
            tmp = Add()([tmp, res])

    md = MDense(pcm_levels, activation='softmax')
    ulaw_prob = md(tmp)
    
    model = Model([pcm, feat], ulaw_prob)
    return model
