import torch
import yaml
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='model checkpoint')
parser.add_argument('setup', type=str, help='setup filename')

if __name__ == "__main__":
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu')

    setup = ckpt['setup']

    with open(args.setup, "w") as f:
        yaml.dump(setup, f)