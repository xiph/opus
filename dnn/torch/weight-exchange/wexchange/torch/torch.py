"""
/* Copyright (c) 2023 Amazon
   Written by Jan Buethe */
/*
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
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import os

import torch
import numpy as np

from wexchange.c_export import CWriter, print_gru_layer2, print_dense_layer, print_conv1d_layer

def dump_torch_gru_weights(where, gru, name='gru', input_sparse=False, recurrent_sparse=False, quantize=False, scale=1/128, recurrent_scale=1/128):

    assert gru.num_layers == 1
    assert gru.bidirectional == False

    w_ih = gru.weight_ih_l0.detach().cpu().numpy()
    w_hh = gru.weight_hh_l0.detach().cpu().numpy()
    b_ih = gru.bias_ih_l0.detach().cpu().numpy()
    b_hh = gru.bias_hh_l0.detach().cpu().numpy()

    if isinstance(where, CWriter):
        return print_gru_layer2(where, name, w_ih, w_hh, b_ih, b_hh, format='torch', input_sparse=input_sparse, recurrent_sparse=recurrent_sparse, quantize=quantize, scale=scale)
    else:
        os.makedirs(where, exist_ok=True)

        np.save(os.path.join(where, 'weight_ih_rzn.npy'), w_ih)
        np.save(os.path.join(where, 'weight_hh_rzn.npy'), w_hh)
        np.save(os.path.join(where, 'bias_ih_rzn.npy'), b_ih)
        np.save(os.path.join(where, 'bias_hh_rzn.npy'), b_hh)



def load_torch_gru_weights(where, gru):

    assert gru.num_layers == 1
    assert gru.bidirectional == False

    w_ih = np.load(os.path.join(where, 'weight_ih_rzn.npy'))
    w_hh = np.load(os.path.join(where, 'weight_hh_rzn.npy'))
    b_ih = np.load(os.path.join(where, 'bias_ih_rzn.npy'))
    b_hh = np.load(os.path.join(where, 'bias_hh_rzn.npy'))

    with torch.no_grad():
        gru.weight_ih_l0.set_(torch.from_numpy(w_ih))
        gru.weight_hh_l0.set_(torch.from_numpy(w_hh))
        gru.bias_ih_l0.set_(torch.from_numpy(b_ih))
        gru.bias_hh_l0.set_(torch.from_numpy(b_hh))


def dump_torch_dense_weights(where, dense, name=None, activation="LINEAR"):

    w = dense.weight.detach().cpu().numpy()
    if dense.bias is None:
        b = np.zeros(dense.out_features, dtype=w.dtype)
    else:
        b = dense.bias.detach().cpu().numpy()

    if isinstance(where, CWriter):
        return print_dense_layer(where, name, w, b, activation, format='torch')

    else:
        os.makedirs(where, exist_ok=True)

        np.save(os.path.join(where, 'weight.npy'), w)
        np.save(os.path.join(where, 'bias.npy'), b)


def load_torch_dense_weights(where, dense):

    w = np.load(os.path.join(where, 'weight.npy'))
    b = np.load(os.path.join(where, 'bias.npy'))

    with torch.no_grad():
        dense.weight.set_(torch.from_numpy(w))
        if dense.bias is not None:
            dense.bias.set_(torch.from_numpy(b))


def dump_torch_conv1d_weights(where, conv, name=None, activation="LINEAR"):

    w = conv.weight.detach().cpu().numpy()
    if conv.bias is None:
        b = np.zeros(conv.out_channels, dtype=w.dtype)
    else:
        b = conv.bias.detach().cpu().numpy()

    if isinstance(where, CWriter):

        return print_conv1d_layer(where, name, w, b, activation, format='torch')
    else:
        os.makedirs(where, exist_ok=True)

        np.save(os.path.join(where, 'weight_oik.npy'), w)

        np.save(os.path.join(where, 'bias.npy'), b)


def load_torch_conv1d_weights(where, conv):

    with torch.no_grad():
        w = np.load(os.path.join(where, 'weight_oik.npy'))
        conv.weight.set_(torch.from_numpy(w))
        if type(conv.bias) != type(None):
            b = np.load(os.path.join(where, 'bias.npy'))
            if conv.bias is not None:
                conv.bias.set_(torch.from_numpy(b))


def dump_torch_embedding_weights(where, emb):
    os.makedirs(where, exist_ok=True)

    w = emb.weight.detach().cpu().numpy()
    np.save(os.path.join(where, 'weight.npy'), w)


def load_torch_embedding_weights(where, emb):

    w = np.load(os.path.join(where, 'weight.npy'))

    with torch.no_grad():
        emb.weight.set_(torch.from_numpy(w))

def dump_torch_weights(where, module, name=None, activation="LINEAR", verbose=False, **kwargs):
    """ generic function for dumping weights of some torch.nn.Module """
    if verbose and name is not None:
        print(f"printing layer {name} of type {type(module)}...")
    if isinstance(module, torch.nn.Linear):
        return dump_torch_dense_weights(where, module, name, activation, **kwargs)
    elif isinstance(module, torch.nn.GRU):
        return dump_torch_gru_weights(where, module, name, **kwargs)
    elif isinstance(module, torch.nn.Conv1d):
        return dump_torch_conv1d_weights(where, module, name, **kwargs)
    elif isinstance(module, torch.nn.Embedding):
        return dump_torch_embedding_weights(where, module)
    else:
        raise ValueError(f'dump_tf_weights: layer of type {type(module)} not supported')

def load_torch_weights(where, module):
    """ generic function for loading weights of some torch.nn.Module """
    if isinstance(module, torch.nn.Linear):
        load_torch_dense_weights(where, module)
    elif isinstance(module, torch.nn.GRU):
        load_torch_gru_weights(where, module)
    elif isinstance(module, torch.nn.Conv1d):
        load_torch_conv1d_weights(where, module)
    elif isinstance(module, torch.nn.Embedding):
        load_torch_embedding_weights(where, module)
    else:
        raise ValueError(f'dump_tf_weights: layer of type {type(module)} not supported')