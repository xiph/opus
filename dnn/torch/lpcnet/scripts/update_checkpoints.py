""" script for updating checkpoints with new setup entries

    Use this script to update older outputs with newly introduced
    parameters. (Saves us the trouble of backward compatibility)
"""


import argparse

import torch

parser = argparse.ArgumentParser()

parser.add_argument('checkpoint_file', type=str, help='checkpoint to be updated')
parser.add_argument('--model', type=str, help='model update', default=None)

args = parser.parse_args()

checkpoint = torch.load(args.checkpoint_file, map_location='cpu')

# update model entry
if type(args.model) != type(None):
    checkpoint['setup']['lpcnet']['model'] = args.model

torch.save(checkpoint, args.checkpoint_file)