from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import activations, initializers, regularizers, constraints, InputSpec, Conv1D, Dense
import numpy as np

class GatedConv(Conv1D):
    
    def __init__(self, filters,
                 kernel_size,
                 dilation_rate=1,
                 activation='tanh',
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

        super(GatedConv, self).__init__(
            filters=2*filters,
            kernel_size=kernel_size,
            strides=1,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation='linear',
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
        self.out_dims = filters
        self.nongate_activation = activations.get(activation)
        
    def call(self, inputs, cond=None, memory=None):
        if memory is None:
            mem = K.zeros((K.shape(inputs)[0], self.mem_size, K.shape(inputs)[-1]))
        else:
            mem = K.variable(K.cast_to_floatx(memory))
        inputs = K.concatenate([mem, inputs], axis=1)
        ret = super(GatedConv, self).call(inputs)
        if cond is not None:
            d = Dense(2*self.out_dims, use_bias=False, activation='linear')
            ret = ret + d(cond)
        ret = self.nongate_activation(ret[:, :, :self.out_dims]) * activations.sigmoid(ret[:, :, self.out_dims:])
        if self.return_memory:
            ret = ret, inputs[:, :self.mem_size, :]
        return ret

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.out_dims
        return tuple(output_shape)
