""" script for updating setup files with new setup entries

    Use this script to update older outputs with newly introduced
    parameters. (Saves us the trouble of backward compatibility)
"""

import argparse

import yaml

parser = argparse.ArgumentParser()

parser.add_argument('setup_file', type=str, help='setup to be updated')
parser.add_argument('--model', type=str, help='model update', default=None)

args = parser.parse_args()

# load setup
with open(args.setup_file, 'r') as f:
    setup = yaml.load(f.read(), yaml.FullLoader)

# update model entry
if type(args.model) != type(None):
    setup['lpcnet']['model'] = args.model

# dump result
with open(args.setup_file, 'w') as f:
    yaml.dump(setup, f)
