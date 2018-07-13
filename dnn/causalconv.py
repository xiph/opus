from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations, initializers, regularizers, constraints, InputSpec, Conv1D
import numpy as np

class CausalConv(Conv1D):
    
    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 return_memory=False,
                 **kwargs):

        super(CausalConv, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.mem_size = dilation_rate*(kernel_size-1)
        self.return_memory = return_memory
        
    def call(self, inputs, memory=None):
        if memory is None:
            mem = K.zeros((K.shape(inputs)[0], self.mem_size, K.shape(inputs)[-1]))
        else:
            mem = K.variable(K.cast_to_floatx(memory))
        inputs = K.concatenate([mem, inputs], axis=1)
        ret = super(CausalConv, self).call(inputs)
        if self.return_memory:
            ret = ret, inputs[:, :self.mem_size, :]
        return ret
