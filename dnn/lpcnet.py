#!/usr/bin/python3

import math
from keras.models import Model
from keras.layers import Input, LSTM, CuDNNGRU, Dense, Embedding, Reshape, Concatenate, Lambda, Conv1D, Multiply, Bidirectional, MaxPooling1D, Activation
from keras import backend as K
from mdense import MDense
import numpy as np
import h5py
import sys

rnn_units=512
pcm_bits = 8
pcm_levels = 2**pcm_bits
nb_used_features = 37


def new_wavernn_model():
    pcm = Input(shape=(None, 1))
    pitch = Input(shape=(None, 1))
    feat = Input(shape=(None, nb_used_features))
    dec_feat = Input(shape=(None, 32))
    dec_state = Input(shape=(rnn_units,))

    conv1 = Conv1D(16, 7, padding='causal')
    pconv1 = Conv1D(16, 5, padding='same')
    pconv2 = Conv1D(16, 5, padding='same')
    fconv1 = Conv1D(128, 3, padding='same')
    fconv2 = Conv1D(32, 3, padding='same')

    if False:
        cpcm = conv1(pcm)
        cpitch = pconv2(pconv1(pitch))
    else:
        cpcm = pcm
        cpitch = pitch

    cfeat = fconv2(fconv1(feat))

    rep = Lambda(lambda x: K.repeat_elements(x, 160, 1))

    rnn = CuDNNGRU(rnn_units, return_sequences=True, return_state=True)
    rnn_in = Concatenate()([cpcm, cpitch, rep(cfeat)])
    md = MDense(pcm_levels, activation='softmax')
    gru_out, state = rnn(rnn_in)
    ulaw_prob = md(gru_out)
    
    model = Model([pcm, pitch, feat], ulaw_prob)
    encoder = Model(feat, cfeat)
    
    dec_rnn_in = Concatenate()([cpcm, cpitch, dec_feat])
    dec_gru_out, state = rnn(dec_rnn_in, initial_state=dec_state)
    dec_ulaw_prob = md(dec_gru_out)

    decoder = Model([pcm, pitch, dec_feat, dec_state], [dec_ulaw_prob, state])
    return model, encoder, decoder
