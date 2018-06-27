#!/usr/bin/python3

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNGRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Bidirectional, MaxPooling1D, Activation
from keras import backend as K
from mdense import MDense
import numpy as np
import h5py
import sys

rnn_units=64
pcm_bits = 8
pcm_levels = 2**pcm_bits
nb_used_features = 37


def new_wavernn_model():
    pcm = Input(shape=(None, 1))
    pitch = Input(shape=(None, 1))
    feat = Input(shape=(None, nb_used_features))

    conv1 = Conv1D(16, 7, padding='causal')
    pconv1 = Conv1D(16, 5, padding='same')
    pconv2 = Conv1D(16, 5, padding='same')
    fconv1 = Conv1D(128, 3, padding='same')
    fconv2 = Conv1D(32, 3, padding='same')

    if True:
        cpcm = conv1(pcm)
        cpitch = pconv2(pconv1(pitch))
    else:
        cpcm = pcm
        cpitch = pitch

    cfeat = fconv2(fconv1(feat))

    rep = Lambda(lambda x: K.repeat_elements(x, 160, 1))

    rnn = CuDNNGRU(rnn_units, return_sequences=True)
    rnn_in = Concatenate()([cpcm, cpitch, rep(cfeat)])
    md = MDense(pcm_levels, activation='softmax')
    ulaw_prob = md(rnn(rnn_in))
    
    model = Model([pcm, pitch, feat], ulaw_prob)
    return model
