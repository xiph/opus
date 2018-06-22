#!/usr/bin/python3

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNGRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Bidirectional, MaxPooling1D, Activation
from keras import backend as K
from mdense import MDense
import numpy as np
import h5py
import sys

rnn_units=256
pcm_bits = 8
pcm_levels = 1+2**pcm_bits

def new_wavernn_model():
    pcm = Input(shape=(None, 1))
    rnn = CuDNNGRU(rnn_units, return_sequences=True)
    md = MDense(pcm_levels, activation='softmax')
    ulaw_prob = md(rnn(pcm))
    
    model = Model(pcm, ulaw_prob)
    return model
