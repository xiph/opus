import argparse

import yaml

from utils.templates import setup_dict

parser = argparse.ArgumentParser()

parser.add_argument('name', type=str, help='name of default setup file')
parser.add_argument('--model', choices=['lpcnet', 'multi_rate'], help='LPCNet model name', default='lpcnet')
parser.add_argument('--path2dataset', type=str, help='dataset path', default=None)

args = parser.parse_args()

setup = setup_dict[args.model]

# update dataset if given
if type(args.path2dataset) != type(None):
    setup['dataset'] = args.path2dataset

name = args.name
if not name.endswith('.yml'):
    name += '.yml'

if __name__ == '__main__':
    with open(name, 'w') as f:
        f.write(yaml.dump(setup))
