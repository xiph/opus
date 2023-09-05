import argparse

import torch
import numpy as np


from models import model_dict
from utils.data import load_features
from utils.wav import wavwrite16

debug = False
if debug:
    args = type('dummy', (object,),
    {
        'features'      : 'features.f32',
        'checkpoint'    : 'checkpoint.pth',
        'output'        : 'out.wav',
        'version'       : 2
    })()
else:
    parser = argparse.ArgumentParser()

    parser.add_argument('features', type=str, help='feature file')
    parser.add_argument('checkpoint', type=str, help='checkpoint file')
    parser.add_argument('output', type=str, help='output file')
    parser.add_argument('--version', type=int, help='feature version', default=2)

    args = parser.parse_args()


torch.set_num_threads(2)

version = args.version
feature_file = args.features
checkpoint_file = args.checkpoint



output_file = args.output
if not output_file.endswith('.wav'):
    output_file += '.wav'

checkpoint = torch.load(checkpoint_file, map_location="cpu")

# check model
if not 'model' in checkpoint['setup']['lpcnet']:
    print(f'warning: did not find model entry in setup, using default lpcnet')
    model_name = 'lpcnet'
else:
    model_name = checkpoint['setup']['lpcnet']['model']

model = model_dict[model_name](checkpoint['setup']['lpcnet']['config'])

model.load_state_dict(checkpoint['state_dict'])

data = load_features(feature_file)

output = model.generate(data['features'], data['periods'], data['lpcs'])

wavwrite16(output_file, output.numpy(), 16000)
