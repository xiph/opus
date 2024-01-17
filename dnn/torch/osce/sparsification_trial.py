import subprocess
import argparse
import sys
import os

import torch
import yaml

from utils.templates import nolace_setup


parser = argparse.ArgumentParser()
parser.add_argument('density', type=float)
parser.add_argument('output', type=str)
parser.add_argument('--pos-offset', type=int, default=1)
parser.add_argument('--num-gpus', type=int, default=torch.cuda.device_count())
parser.add_argument('--dataset-path', type=str, default=None)
parser.add_argument('--testdata', type=str, default=None)


if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    setup = nolace_setup
    procs = []
    if args.dataset_path is not None:
        setup['dataset'] = os.path.join(args.dataset_path, 'training')
        setup['validation_dataset'] = os.path.join(args.dataset_path, 'validation')

    setup['model']['kwargs']['sparsify'] = True
    for cuda_idx in range(args.num_gpus):
        densities = 9*[1]
        densities[cuda_idx+args.pos_offset] = args.density
        setup['model']['kwargs']['sparsification_density'] = densities
        setup['model']['kwargs']['sparsification_schedule'] = [10000, 30000, 100]

        output_folder = os.path.join(args.output, f"nolace_d{args.density}_p{cuda_idx+args.pos_offset}")
        setup_path = os.path.join(args.output, f"setup_d{args.density}_p{cuda_idx+args.pos_offset}.yml")
        with open(setup_path, "w") as f:
            f.write(yaml.dump(setup))

        trainmodel = os.path.join(os.path.split(__file__)[0], "train_model.py")
        cmd = [sys.executable, trainmodel, setup_path, output_folder, "--device", f"cuda:{cuda_idx}"]
        if args.testdata is not None:
            cmd += ['--testdata', args.testdata]

        procs.append(subprocess.Popen(" ".join(cmd), shell=True))
        print(procs[-1].pid)
