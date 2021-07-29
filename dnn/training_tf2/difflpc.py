"""
Tensorflow model (differentiable lpc) to learn the lpcs from the features
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Conv1D, Multiply, Layer, LeakyReLU
from tensorflow.keras import backend as K
from tf_funcs import diff_rc2lpc

frame_size = 160
lpcoeffs_N = 16

def difflpc(nb_used_features = 20, training=False):
    feat = Input(shape=(None, nb_used_features)) # BFCC
    padding = 'valid' if training else 'same'
    L1 = Conv1D(100, 3, padding=padding, activation='tanh', name='f2rc_conv1')
    L2 = Conv1D(75, 3, padding=padding, activation='tanh', name='f2rc_conv2')
    L3 = Dense(50, activation='tanh',name = 'f2rc_dense3')
    L4 = Dense(lpcoeffs_N, activation='tanh',name = "f2rc_dense4_outp_rc")
    rc = L4(L3(L2(L1(feat))))
    # Differentiable RC 2 LPC
    lpcoeffs = diff_rc2lpc(name = "rc2lpc")(rc)

    model = Model(feat,lpcoeffs,name = 'f2lpc')
    model.nb_used_features = nb_used_features
    model.frame_size = frame_size
    return model
