import argparse

import yaml

from models import model_dict


debug = False
if debug:
    args = type('dummy', (object,),
    {
        'setup' : 'setups/lpcnet_m/setup_1_4_concatenative.yml',
        'hierarchical_sampling' : False
    })()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('setup', type=str, help='setup yaml file')
    parser.add_argument('--hierarchical-sampling', action="store_true", help='whether to assume hierarchical sampling (default=False)', default=False)

    args = parser.parse_args()

with open(args.setup, 'r') as f:
    setup = yaml.load(f.read(), yaml.FullLoader)

# check model
if not 'model' in setup['lpcnet']:
    print(f'warning: did not find model entry in setup, using default lpcnet')
    model_name = 'lpcnet'
else:
    model_name = setup['lpcnet']['model']

# create model
model = model_dict[model_name](setup['lpcnet']['config'])

gflops = model.get_gflops(16000, verbose=True, hierarchical_sampling=args.hierarchical_sampling)
