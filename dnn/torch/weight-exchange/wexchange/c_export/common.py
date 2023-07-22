'''Copyright (c) 2017-2018 Mozilla
   Copyright (c) 2022 Amazon

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
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np

from .c_writer import CWriter

def print_vector(writer, vector, name, dtype='float', dotp=False, static=True):

    f = writer.source
    binary_blob = writer.enable_binary_blob

    if binary_blob:
        f.write(
f'''
#ifndef USE_WEIGHTS_FILE
#define WEIGHTS_{name}_DEFINED
#define WEIGHTS_{name}_TYPE WEIGHT_TYPE_{"qweight" if dotp else "float"}
'''
        )
        writer.weight_arrays.add(name)

    if dotp:
        vector = vector.reshape((vector.shape[0]//4, 4, vector.shape[1]//8, 8))
        vector = vector.transpose((2, 0, 3, 1))

    v = np.reshape(vector, (-1))

    if static:
        f.write('static ')

    f.write(f'const {dtype} {name}[{len(v)}] = {{\n    ')

    for i in range(0, len(v)):

        f.write(f'{v[i]}')

        if (i!=len(v)-1):
            f.write(',')
        else:
            break

        if (i%8==7):
            f.write("\n    ")
        else:
            f.write(" ")

    f.write('\n};\n\n')
    if binary_blob:
        f.write(
f'''
#endif /* USE_WEIGHTS_FILE */
'''
        )

    return vector



def print_sparse_vector(writer, A, name, have_diag=True):
    f = writer.source
    N = A.shape[0]
    M = A.shape[1]
    W = np.zeros((0,), dtype='int')
    W0 = np.zeros((0,))
    if have_diag:
        diag = np.concatenate([np.diag(A[:,:N]), np.diag(A[:,N:2*N]), np.diag(A[:,2*N:])])
        A[:,:N] = A[:,:N] - np.diag(np.diag(A[:,:N]))
        A[:,N:2*N] = A[:,N:2*N] - np.diag(np.diag(A[:,N:2*N]))
        A[:,2*N:] = A[:,2*N:] - np.diag(np.diag(A[:,2*N:]))
        print_vector(writer, diag, name + '_diag')
    AQ = np.minimum(127, np.maximum(-128, np.round(A*128))).astype('int')
    idx = np.zeros((0,), dtype='int')
    for i in range(M//8):
        pos = idx.shape[0]
        idx = np.append(idx, -1)
        nb_nonzero = 0
        for j in range(N//4):
            block = A[j*4:(j+1)*4, i*8:(i+1)*8]
            qblock = AQ[j*4:(j+1)*4, i*8:(i+1)*8]
            if np.sum(np.abs(block)) > 1e-10:
                nb_nonzero = nb_nonzero + 1
                idx = np.append(idx, j*4)
                vblock = qblock.transpose((1,0)).reshape((-1,))
                W0 = np.concatenate([W0, block.reshape((-1,))])
                W = np.concatenate([W, vblock])
        idx[pos] = nb_nonzero
    f.write('#ifdef DOT_PROD\n')
    print_vector(writer, W, name, dtype='qweight')
    f.write('#else /*DOT_PROD*/\n')
    print_vector(writer, W0, name, dtype='qweight')
    f.write('#endif /*DOT_PROD*/\n')

    print_vector(writer, idx, name + '_idx', dtype='int')
    return AQ

def _check_activation(activation):
    if not activation in {"TANH", "SIGMOID", "LINEAR", "SWISH", "RELU", "SOFTMAX"}:
        raise ValueError(f"error: unknown activation {activation}")

def print_dense_layer(writer : CWriter,
                      name : str,
                      weight : np.ndarray,
                      bias : np.ndarray,
                      activation: str,
                      format : str = 'torch'):

    _check_activation(activation)

    if format == 'torch':
        weight = weight.transpose()

    print_vector(writer, weight, name + "_weights")
    print_vector(writer, bias, name + "_bias")

    writer.header.write(f"\n#define {name.upper()}_OUT_SIZE {weight.shape[1]}\n")

    if writer.enable_binary_blob:
        init_call = f'dense_init(&model->{name}, arrays, "{name}_bias", "{name}_weights", {weight.shape[0]}, {weight.shape[1]}, ACTIVATION_{activation})'
        writer.layer_dict[name] = ('DenseLayer', init_call)
    else:
        writer.source.write(
f"""

const DenseLayer {name} = {{
   {name}_bias,
   {name}_weights,
   {weight.shape[0]},
   {weight.shape[1]},
   ACTIVATION_{activation}
}};

"""
        )

        writer.header.write(f"\nextern const DenseLayer {name};\n\n")





def print_conv1d_layer(writer : CWriter,
                       name : str,
                       weight : np.ndarray,
                       bias : np.ndarray,
                       activation: str,
                       format : str = 'torch'):

    _check_activation(activation)

    if format == "torch":
        # convert to channels last
        weight = np.transpose(weight, (2, 1, 0))

    print_vector(writer, weight, name + "_weights")
    print_vector(writer, bias, name + "_bias")

    writer.header.write(f"\n#define {name.upper()}_OUT_SIZE {weight.shape[2]}\n")
    writer.header.write(f"\n#define {name.upper()}_STATE_SIZE ({weight.shape[1]} * ({weight.shape[0] - 1}))\n")
    writer.header.write(f"\n#define {name.upper()}_DELAY {(weight.shape[0] - 1) // 2}\n") # CAVE: delay is not a property of the conv layer

    if writer.enable_binary_blob:
        init_call = f'conv1d_init(&model->{name}, arrays, "{name}_bias", "{name}_weights", {weight.shape[1]}, {weight.shape[0]}, {weight.shape[2]}, ACTIVATION_{activation})'
        writer.layer_dict[name] = ('Conv1DLayer', init_call)
    else:

        writer.source.write(
f"""

const Conv1DLayer {name} = {{
   {name}_bias,
   {name}_weights,
   {weight.shape[1]},
   {weight.shape[0]},
   {weight.shape[2]},
   ACTIVATION_{activation}
}};

"""
        )

        writer.header.write(f"\nextern const Conv1DLayer {name};\n\n")

    return weight.shape[0] * weight.shape[1]


def print_gru_layer(writer : CWriter,
                    name : str,
                    weight : np.ndarray,
                    recurrent_weight : np.ndarray,
                    bias : np.ndarray,
                    recurrent_bias : np.ndarray,
                    activation: str,
                    format : str = 'torch',
                    dotp : bool = False,
                    input_sparse : bool = False,
                    reset_after : int = 0
                    ):

    _check_activation(activation)

    if format == "torch":
        # transpose weight matrices and change gate order from rzn to zrn

        N = weight.shape[0] // 3
        for x in [weight, recurrent_weight, bias, recurrent_bias]:
            tmp = x[0:N].copy()
            x[0:N] = x[N:2*N]
            x[N:2*N] = tmp

        weight = weight.transpose()
        recurrent_weight = recurrent_weight.transpose()


    # input weights
    if input_sparse:
        qweight = print_sparse_vector(writer, weight, name + '_weights', have_diag=False)
    else:
        qweight = np.clip(np.round(128. * weight).astype('int'), -128, 127)

        if dotp:
            writer.source.write("#ifdef DOT_PROD\n")
            print_vector(writer, qweight, name + '_weights', dtype='qweight', dotp=True)
            writer.source.write("#else /*DOT_PROD*/\n")

        print_vector(writer, weight, name + '_weights')

        if dotp:
             writer.source.write("#endif /*DOT_PROD*/\n")


    # recurrent weights
    recurrent_qweight = np.clip(np.round(128. * recurrent_weight).astype('int'), -128, 127)

    if dotp:
        writer.source.write("#ifdef DOT_PROD\n")
        print_vector(writer, recurrent_qweight, name + '_recurrent_weights', dtype='qweight', dotp=True)
        writer.source.write("#else /*DOT_PROD*/\n")

    print_vector(writer, recurrent_weight, name + '_recurrent_weights')

    if dotp:
        writer.source.write("#endif /*DOT_PROD*/\n")


    # corrected bias for unsigned int matrix multiplication
    subias              = bias - np.sum(qweight / 128., axis=0)
    recurrent_subias    = recurrent_bias - np.sum(recurrent_qweight / 128., axis=0)

    print_vector(writer, np.concatenate((bias, recurrent_bias)), name + "_bias")
    print_vector(writer, np.concatenate((subias, recurrent_subias)), name + "_subias")


    # wrapping it up
    writer.header.write(f"\n#define {name.upper()}_OUT_SIZE {N}\n")
    writer.header.write(f"\n#define {name.upper()}_STATE_SIZE {N}\n")

    if writer.enable_binary_blob:
        if input_sparse:
            init_call = f'gru_init(&model->{name}, arrays, "{name}_bias", "{name}_subias", "{name}_weights", "{name + "_weights_idx"}", "{name}_recurrent_weights", {weight.shape[0]}, {weight.shape[1] // 3}, ACTIVATION_{activation}, {reset_after})'
        else:
            init_call = f'gru_init(&model->{name}, arrays, "{name}_bias", "{name}_subias", "{name}_weights", NULL, "{name}_recurrent_weights", {weight.shape[0]}, {weight.shape[1] // 3}, ACTIVATION_{activation}, {reset_after})'

        writer.layer_dict[name] = ('GRULayer', init_call)

    else:

        writer.source.write(
f"""

const GRULayer {name} = {{
   {name}_bias,
   {name}_subias,
   {name}_weights,
   {name + "_weights_idx" if input_sparse else "NULL"},
   {name}_recurrent_weights,
   {weight.shape[0]},
   {weight.shape[1] // 3},
   ACTIVATION_{activation},
   {reset_after}
}};

"""
        )

        writer.header.write(f"\nextern const GRULayer {name};\n")


    return N


