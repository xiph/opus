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
import argparse
import sys

import hashlib

sys.path.append(os.path.join(os.path.dirname(__file__), '../weight-exchange'))

import torch
import wexchange.torch
from wexchange.torch import dump_torch_weights
from models import model_dict

parser = argparse.ArgumentParser()

parser.add_argument('checkpoint', type=str, help='LACE or NoLACE model checkpoint')
parser.add_argument('output_dir', type=str, help='output folder')


# auxiliary functions
def sha1(filename):
    BUF_SIZE = 65536
    sha1 = hashlib.sha1()

    with open(filename, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest()

def export_name(name):
    return name.replace('.', '_')

if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    outdir = args.output_dir
    os.makedirs(outdir, exist_ok=True)

    # dump message
    message = f"Auto generated from checkpoint {os.path.basename(checkpoint_path)} (sha1: {sha1(checkpoint_path)})"

    # create model and load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = model_dict[checkpoint['setup']['model']['name']](*checkpoint['setup']['model']['args'], **checkpoint['setup']['model']['kwargs'])

    # CWriter
    model_name = checkpoint['setup']['model']['name']
    cwriter = wexchange.c_export.CWriter(os.path.join(outdir, model_name + "_data"), message=message, model_struct_name=model_name.upper())

    # dump numbits_embedding parameters by hand
    numbits_embedding = model.get_submodule('numbits_embedding')
    weights = next(iter(numbits_embedding.parameters()))
    for i, c in enumerate(weights):
        cwriter.header.write(f"\nNUMBITS_COEF_{i} {float(c.detach())}f")
    cwriter.header.write("\n\n")

    # dump layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv1d) \
            or isinstance(module, torch.nn.ConvTranspose1d) or isinstance(module, torch.nn.Embedding):
                dump_torch_weights(cwriter, module, name=export_name(name), verbose=True)

    cwriter.close()
