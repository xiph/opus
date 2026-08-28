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

"""Distort LACE, NoLACE and BBWENet weights to a target SNR and regenerate the
opus/dnn/<model>_data.c weight files.

Additive white Gaussian noise is applied per weight tensor so that each tensor
reaches the requested signal-to-noise ratio (in dB):

    noise_power = mean(weight ** 2) / 10 ** (snr_db / 10)

Only trainable weight matrices (tensors with >= 2 dimensions, i.e. Linear / Conv
/ GRU / Embedding weights) are distorted; biases, other 1-D parameters and fixed
non-learned buffers (e.g. the TDShaper interpolation kernels) are left untouched.
Distortion is applied to the effective weights after weight-norm has been removed,
i.e. to exactly the values that get written to the C files, and the regenerated
files are produced by the regular export pipeline so that the int8 / float / scale
arrays stay mutually consistent.
"""

import os
import sys
import argparse

import torch

# make the osce package and the weight-exchange package importable
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, '..', '..'))                              # dnn/torch/osce
sys.path.append(os.path.join(file_dir, '..', '..', '..', 'weight-exchange'))    # dnn/torch/weight-exchange
sys.path.append(os.path.join(file_dir, '..', '..', '..', 'dnntools'))           # dnn/torch/dnntools

# opus/dnn, i.e. the directory holding the committed <model>_data.c files and models/
DNN_DIR = os.path.abspath(os.path.join(file_dir, '..', '..', '..', '..'))
DEFAULT_MODELS_DIR = os.path.join(DNN_DIR, 'models')
DEFAULT_OUTPUT_DIR = DNN_DIR

# checkpoints in dnn/models used to regenerate each model's weight file
CHECKPOINTS = {
    'lace':    'lace_v2.pth',
    'nolace':  'nolace_160_v2.pth',
    'bbwenet': 'bbwenet_v2.pth',
}

# per-model seed offset so that distorting a single model gives the same result
# whether or not the other models are distorted in the same run
SEED_OFFSETS = {'lace': 0, 'nolace': 1, 'bbwenet': 2}


def make_distortion(snr_db, generator):
    """Return a transform(model) that adds AWGN to weight matrices at the given SNR (dB)."""
    ratio = 10.0 ** (snr_db / 10.0)

    def distort(model):
        num_tensors = 0
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.dim() < 2 or not param.requires_grad:
                    # skip biases / 1-D params and fixed (non-learned) buffers such as
                    # the TDShaper interpolation kernels
                    continue
                signal_power = param.pow(2).mean()
                if float(signal_power) == 0.0:
                    continue
                noise_std = (signal_power / ratio).sqrt()
                noise = torch.randn(param.shape, generator=generator, dtype=param.dtype) * noise_std
                param.add_(noise)
                num_tensors += 1
        print(f"    distorted {num_tensors} weight tensors at SNR = {snr_db} dB")

    return distort


def main():
    parser = argparse.ArgumentParser(
        description="Distort LACE/NoLACE/BBWENet weights to a target per-tensor SNR and "
                    "regenerate the opus/dnn/<model>_data.c files."
    )
    parser.add_argument('--snr', type=float, required=True,
                        help='target per-tensor weight SNR in dB (lower means more distortion)')
    parser.add_argument('--seed', type=int, default=0,
                        help='base random seed for reproducible noise (default: 0)')
    parser.add_argument('--models', nargs='+', choices=list(CHECKPOINTS.keys()),
                        default=list(CHECKPOINTS.keys()),
                        help='models to distort (default: all)')
    parser.add_argument('--models-dir', type=str, default=DEFAULT_MODELS_DIR,
                        help=f'directory containing the .pth checkpoints (default: {DEFAULT_MODELS_DIR})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'directory to write <model>_data.{{c,h}} (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--no-quantize', dest='quantize', action='store_false',
                        help='disable the quantization schedule (the committed files use quantization)')
    parser.set_defaults(quantize=True)

    args = parser.parse_args()

    import export_model_weights as emw

    for model_name in args.models:
        checkpoint_path = os.path.join(args.models_dir, CHECKPOINTS[model_name])
        output_path = os.path.join(args.output_dir, model_name + '_data.c')

        if not os.path.isfile(checkpoint_path):
            print(f"[{model_name}] skip: checkpoint not found at {checkpoint_path}")
            continue

        seed = args.seed + SEED_OFFSETS[model_name]
        print(f"[{model_name}] {CHECKPOINTS[model_name]} -> {output_path} (SNR = {args.snr} dB, seed = {seed})")

        generator = torch.Generator().manual_seed(seed)
        transform = make_distortion(args.snr, generator)

        emw.export_model(checkpoint_path, args.output_dir, quantize=args.quantize,
                         transform=transform, strict=False)

    print("done")


if __name__ == "__main__":
    main()
