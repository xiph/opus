import os

import tensorflow as tf
import numpy as np

from wexchange.c_export import CWriter, print_gru_layer, print_dense_layer, print_conv1d_layer

def dump_tf_gru_weights(where, gru, name=None, input_sparse=False, dotp=False):


    assert gru.activation == tf.keras.activations.tanh
    assert gru.recurrent_activation == tf.keras.activations.sigmoid
    assert gru.reset_after == True

    w_ih = gru.weights[0].numpy().transpose().copy()
    w_hh = gru.weights[1].numpy().transpose().copy()
    b_ih = gru.weights[2].numpy()[0].copy()
    b_hh = gru.weights[2].numpy()[1].copy()

    if isinstance(where, CWriter):
        return print_gru_layer(where, name, w_ih, w_hh, b_ih, b_hh, 'TANH', format='tf', reset_after=1, input_sparse=input_sparse, dotp=dotp)
    else:
        os.makedirs(where, exist_ok=True)

        # zrn => rzn
        N = w_ih.shape[0] // 3
        for x in [w_ih, w_hh, b_ih, b_hh]:
            tmp = x[0:N].copy()
            x[0:N] = x[N:2*N]
            x[N:2*N] = tmp

        np.save(os.path.join(where, 'weight_ih_rzn.npy'), w_ih)
        np.save(os.path.join(where, 'weight_hh_rzn.npy'), w_hh)
        np.save(os.path.join(where, 'bias_ih_rzn.npy'), b_ih)
        np.save(os.path.join(where, 'bias_hh_rzn.npy'), b_hh)


def load_tf_gru_weights(path, gru):

    assert gru.activation == tf.keras.activations.tanh
    assert gru.recurrent_activation == tf.keras.activations.sigmoid
    assert gru.reset_after == True

    w_ih = np.load(os.path.join(path, 'weight_ih_rzn.npy'))
    w_hh = np.load(os.path.join(path, 'weight_hh_rzn.npy'))
    b_ih = np.load(os.path.join(path, 'bias_ih_rzn.npy'))
    b_hh = np.load(os.path.join(path, 'bias_hh_rzn.npy'))

    # rzn => zrn
    N = w_ih.shape[0] // 3
    for x in [w_ih, w_hh, b_ih, b_hh]:
        tmp = x[0:N].copy()
        x[0:N] = x[N:2*N]
        x[N:2*N] = tmp

    gru.weights[0].assign(tf.convert_to_tensor(w_ih.transpose()))
    gru.weights[1].assign(tf.convert_to_tensor(w_hh.transpose()))
    gru.weights[2].assign(tf.convert_to_tensor(np.vstack((b_ih, b_hh))))


def dump_tf_dense_weights(where, dense, name=None):

    w = dense.weights[0].numpy()
    if dense.bias is None:
        b = np.zeros(dense.units, dtype=w.dtype)
    else:
        b = dense.bias.numpy()



    if isinstance(where, CWriter):
        try:
            activation = dense.activation.__name__.upper()
        except:
            activation = "LINEAR"

        return print_dense_layer(where, name, w, b, activation, format='tf')

    else:
        os.makedirs(where, exist_ok=True)

        np.save(os.path.join(where, 'weight.npy'), w.transpose())
        np.save(os.path.join(where, 'bias.npy'), b)


def load_tf_dense_weights(path, dense):

    w = np.load(os.path.join(path, 'weight.npy')).transpose()
    b = np.load(os.path.join(path, 'bias.npy'))

    dense.weights[0].assign(tf.convert_to_tensor(w))
    if dense.bias is not None:
        dense.weights[1].assign(tf.convert_to_tensor(b))


def dump_tf_conv1d_weights(where, conv, name=None):

    assert conv.data_format == 'channels_last'

    w = conv.weights[0].numpy().copy()
    if conv.bias is None:
        b = np.zeros(conv.filters, dtype=w.dtype)
    else:
        b = conv.bias.numpy()

    if isinstance(where, CWriter):
        try:
            activation = conv.activation.__name__.upper()
        except:
            activation = "LINEAR"

        return print_conv1d_layer(where, name, w, b, activation, format='tf')
    else:
        os.makedirs(where, exist_ok=True)

        w = np.transpose(w, (2, 1, 0))
        np.save(os.path.join(where, 'weight_oik.npy'), w)
        np.save(os.path.join(where, 'bias.npy'), b)


def load_tf_conv1d_weights(path, conv):

    w = np.load(os.path.join(path, 'weight_oik.npy'))
    b = np.load(os.path.join(path, 'bias.npy'))

    w = np.transpose(w, (2, 1, 0))

    conv.weights[0].assign(tf.convert_to_tensor(w))
    if conv.bias is not None:
        conv.weights[1].assign(tf.convert_to_tensor(b))


def dump_tf_embedding_weights(path, emb):
    os.makedirs(path, exist_ok=True)

    w = emb.weights[0].numpy()
    np.save(os.path.join(path, 'weight.npy'), w)



def load_tf_embedding_weights(path, emb):

    w = np.load(os.path.join(path, 'weight.npy'))
    emb.weights[0].assign(tf.convert_to_tensor(w))


def dump_tf_weights(path, module):
    if isinstance(module, tf.keras.layers.Dense):
        dump_tf_dense_weights(path, module)
    elif isinstance(module, tf.keras.layers.GRU):
        dump_tf_gru_weights(path, module)
    elif isinstance(module, tf.keras.layers.Conv1D):
        dump_tf_conv1d_weights(path, module)
    elif isinstance(module, tf.keras.layers.Embedding):
        dump_tf_embedding_weights(path, module)
    else:
        raise ValueError(f'dump_tf_weights: layer of type {type(module)} not supported')

def load_tf_weights(path, module):
    if isinstance(module, tf.keras.layers.Dense):
        load_tf_dense_weights(path, module)
    elif isinstance(module, tf.keras.layers.GRU):
        load_tf_gru_weights(path, module)
    elif isinstance(module, tf.keras.layers.Conv1D):
        load_tf_conv1d_weights(path, module)
    elif isinstance(module, tf.keras.layers.Embedding):
        load_tf_embedding_weights(path, module)
    else:
        raise ValueError(f'dump_tf_weights: layer of type {type(module)} not supported')