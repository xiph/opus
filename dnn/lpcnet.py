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
    feat = Input(shape=(None, nb_used_features))

    rep = Lambda(lambda x: K.repeat_elements(x, 160, 1))

    rnn = CuDNNGRU(rnn_units, return_sequences=True)
    rnn_in = Concatenate()([pcm, rep(feat)])
    md = MDense(pcm_levels, activation='softmax')
    ulaw_prob = md(rnn(rnn_in))
    
    model = Model([pcm, feat], ulaw_prob)
    return model
