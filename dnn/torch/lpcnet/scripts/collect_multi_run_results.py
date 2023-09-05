import argparse
import os
from uuid import UUID
from collections import OrderedDict
import pickle


import torch
import numpy as np

import utils



parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="input folder containing multi-run output")
parser.add_argument("tag", type=str, help="tag for multi-run experiment")
parser.add_argument("csv", type=str, help="name for output csv")


def is_uuid(val):
    try:
        UUID(val)
        return True
    except:
        return False


def collect_results(folder):

    training_folder = os.path.join(folder, 'training')
    testing_folder  = os.path.join(folder, 'testing')

    # validation loss
    checkpoint = torch.load(os.path.join(training_folder, 'checkpoints', 'checkpoint_finalize_epoch_1.pth'), map_location='cpu')
    validation_loss = checkpoint['validation_loss']

    # eval_warpq
    eval_warpq = utils.data.parse_warpq_scores(os.path.join(training_folder, 'out_finalize.txt'))[-1]

    # testing results
    testing_results = utils.data.collect_test_stats(os.path.join(testing_folder, 'final'))

    results = OrderedDict()
    results['eval_loss']          = validation_loss
    results['eval_warpq']         = eval_warpq
    results['pesq_mean']          = testing_results['pesq'][0]
    results['warpq_mean']         = testing_results['warpq'][0]
    results['pitch_error_mean']   = testing_results['pitch_error'][0]
    results['voicing_error_mean'] = testing_results['voicing_error'][0]

    return results

def print_csv(path, results, tag, ranks=None, header=True):

    metrics = next(iter(results.values())).keys()
    if ranks is not None:
        rank_keys = next(iter(ranks.values())).keys()
    else:
        rank_keys = []

    with open(path, 'w') as f:
        if header:
            f.write("uuid, tag")

            for metric in metrics:
                f.write(f", {metric}")

            for rank in rank_keys:
                f.write(f", {rank}")

            f.write("\n")


        for uuid, values in results.items():
            f.write(f"{uuid}, {tag}")

            for val in values.values():
                f.write(f", {val:10.8f}")

            for rank in rank_keys:
                f.write(f", {ranks[uuid][rank]:4d}")

            f.write("\n")

def get_ranks(results):

    metrics = list(next(iter(results.values())).keys())

    positive = {'pesq_mean', 'mix'}

    ranks = OrderedDict()
    for key in results.keys():
        ranks[key] = OrderedDict()

    for metric in metrics:
        sign = -1 if metric in positive else 1

        x = sorted([(key, value[metric]) for key, value in results.items()], key=lambda x: sign * x[1])
        x = [y[0] for y in x]

        for key in results.keys():
            ranks[key]['rank_' + metric] = x.index(key) + 1

    return ranks

def analyse_metrics(results):
    metrics = ['eval_loss', 'pesq_mean', 'warpq_mean', 'pitch_error_mean', 'voicing_error_mean']

    x = []
    for metric in metrics:
        x.append([val[metric] for val in results.values()])

    x = np.array(x)

    print(x)

def add_mix_metric(results):
    metrics = ['eval_loss', 'pesq_mean', 'warpq_mean', 'pitch_error_mean', 'voicing_error_mean']

    x = []
    for metric in metrics:
        x.append([val[metric] for val in results.values()])

    x = np.array(x).transpose() * np.array([-1, 1, -1, -1, -1])

    z = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    print(f"covariance matrix for normalized scores of {metrics}:")
    print(np.cov(z.transpose()))

    score = np.mean(z, axis=1)

    for i, key in enumerate(results.keys()):
        results[key]['mix'] = score[i].item()

if __name__ == "__main__":
    args = parser.parse_args()

    uuids = sorted([x for x in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, x)) and is_uuid(x)])


    results = OrderedDict()

    for uuid in uuids:
        results[uuid] = collect_results(os.path.join(args.input, uuid))


    add_mix_metric(results)

    ranks = get_ranks(results)



    csv = args.csv if args.csv.endswith('.csv') else args.csv + '.csv'

    print_csv(args.csv, results, args.tag, ranks=ranks)


    with open(csv[:-4] + '.pickle', 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)