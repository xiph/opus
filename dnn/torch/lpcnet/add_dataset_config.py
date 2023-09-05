import argparse
import os

import yaml


from utils.templates import dataset_template_v1, dataset_template_v2




parser = argparse.ArgumentParser("add_dataset_config.py")

parser.add_argument('path', type=str, help='path to folder containing feature and data file')
parser.add_argument('--version', type=int, help="dataset version, 1 for classic LPCNet with 55 feature slots, 2 for new format with 36 feature slots.", default=2)
parser.add_argument('--description', type=str, help='brief dataset description', default="I will add a description later")
args = parser.parse_args()


if args.version == 1:
    template = dataset_template_v1
    data_extension = '.u8'
elif args.version == 2:
    template = dataset_template_v2
    data_extension = '.s16'
else:
    raise ValueError(f"unknown dataset version {args.version}")

# get folder content
content = os.listdir(args.path)

features = [c for c in content if c.endswith('.f32')]

if len(features) != 1:
    print("could not determine feature file")
else:
    template['feature_file'] = features[0]

data = [c for c in content if c.endswith(data_extension)]
if len(data) != 1:
    print("could not determine data file")
else:
    template['signal_file'] = data[0]

template['description'] = args.description

with open(os.path.join(args.path, 'info.yml'), 'w') as f:
    yaml.dump(template, f)
