import os
import argparse

import torch
import numpy as np

from models import model_dict
from utils import endoscopy

parser = argparse.ArgumentParser()

parser.add_argument('checkpoint_path', type=str, help='path to folder containing checkpoints "lace_checkpoint.pth" and nolace_checkpoint.pth"')
parser.add_argument('output_folder', type=str, help='output folder for testvectors')
parser.add_argument('--debug', action='store_true', help='add debug output to output folder')


def create_adaconv_testvector(prefix, adaconv, num_frames, debug=False):
    feature_dim = adaconv.feature_dim
    in_channels = adaconv.in_channels
    frame_size = adaconv.frame_size

    features = torch.randn((1, num_frames, feature_dim))
    x_in = torch.randn((1, in_channels, num_frames * frame_size))

    x_out = adaconv(x_in, features, debug=debug)

    features = features[0].detach().numpy()
    x_in = x_in[0].permute(1, 0).detach().numpy()
    x_out = x_out[0].permute(1, 0).detach().numpy()

    features.tofile(prefix + '_features.f32')
    x_in.tofile(prefix + '_x_in.f32')
    x_out.tofile(prefix + '_x_out.f32')

if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    lace_checkpoint = torch.load(os.path.join(args.checkpoint_path, "lace_checkpoint.pth"), map_location='cpu')
    nolace_checkpoint = torch.load(os.path.join(args.checkpoint_path, "nolace_checkpoint.pth"), map_location='cpu')

    lace = model_dict['lace'](**lace_checkpoint['setup']['model']['kwargs'])
    nolace = model_dict['nolace'](**nolace_checkpoint['setup']['model']['kwargs'])

    lace.load_state_dict(lace_checkpoint['state_dict'])
    nolace.load_state_dict(nolace_checkpoint['state_dict'])

    if args.debug:
        endoscopy.init(args.output_folder)

    # lace af1, 1 input channel, 1 output channel
    create_adaconv_testvector(os.path.join(args.output_folder, "lace_af1"), lace.af1, 5, debug=args.debug)


    # nolace af1, 1 input channel, 2 output channels
    create_adaconv_testvector(os.path.join(args.output_folder, "nolace_af1"), nolace.af1, 5, debug=args.debug)

    # nolace af4, 2 input channel, 1 output channels
    create_adaconv_testvector(os.path.join(args.output_folder, "nolace_af4"), nolace.af4, 5, debug=args.debug)

    # nolace af2, 2 input channel, 2 output channels
    create_adaconv_testvector(os.path.join(args.output_folder, "nolace_af2"), nolace.af2, 5, debug=args.debug)

    if args.debug:
        endoscopy.close()
