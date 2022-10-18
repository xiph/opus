import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument('weights', metavar="<weight file>", type=str, help='model weight file in hdf5 format')
parser.add_argument('--cond-size', type=int, help="conditioning size (default: 256)", default=256)
parser.add_argument('--latent-dim', type=int, help="dimension of latent space (default: 80)", default=80)

args = parser.parse_args()

# now import the heavy stuff
from keraslayerdump import dump_conv1d_layer, dump_dense_layer, dump_gru_layer
from rdovae import new_rdovae_model

def start_header(header_fid, header_name):
    header_guard = "_" + os.path.basename(header_name)[:-2].upper() + "_H"
    header_fid.write(
f"""
#ifndef {header_guard}
#define {header_guard}

#include "nnet.h"

"""
    )

def finish_header(header_fid):
    header_fid.write(
"""
#endif

"""
    )

def start_source(source_fid, header_name, weight_file):
    source_fid.write(
f"""
/* this source file was automatically generated from weight file {weight_file} */

#include "{header_name}"

"""
    )

def finish_source(source_fid):
    pass


if __name__ == "__main__":

    model, encoder, decoder, qembedding = new_rdovae_model(20, args.latent_dim, cond_size=args.cond_size)
    model.load_weights(args.weights)


    # for the time being only dump encoder
    encoder_dense_names = [
        'enc_dense1',
        'enc_dense3',
        'enc_dense5',
        'enc_dense7',
        'enc_dense8',
        'gdense1',
        'gdense2'
    ]

    encoder_gru_names = [
        'enc_dense2',
        'enc_dense4',
        'enc_dense6'
    ]

    encoder_conv1d_names = [
        'bits_dense'
    ]

    source_fid = open("nfec_enc_data.c", 'w')
    header_fid = open("nfec_enc_data.h", 'w')

    start_header(header_fid, "nfec_enc_data.h")
    start_source(source_fid, "nfec_enc_data.h", os.path.basename(args.weights))

    # dump GRUs
    max_rnn_neurons = max(
        [
            dump_gru_layer(encoder.get_layer(name), source_fid, header_fid)
            for name in encoder_gru_names
        ]
    )

    # dump conv layers
    max_conv_inputs = max(
        [
            dump_conv1d_layer(encoder.get_layer(name), source_fid, header_fid)
            for name in encoder_conv1d_names
        ] 
    )

    # dump Dense layers
    for name in encoder_dense_names:
        layer = encoder.get_layer(name)
        dump_dense_layer(layer, source_fid, header_fid)

    # some global constants
    header_fid.write(
f"""
#define NFEC_NUM_FEATURES 20

#define NFEC_LATENT_DIM {args.latent_dim}

#define NFEC_ENC_MAX_RNN_NEURONS {max_rnn_neurons}

#define NFEC_ENC_MAX_CONV_INPUTS {max_conv_inputs}

"""
    )

    finish_header(header_fid)
    finish_source(source_fid)

    header_fid.close()
    source_fid.close()

