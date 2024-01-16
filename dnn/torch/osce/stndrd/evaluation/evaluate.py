import os
import argparse


from scipy.io import wavfile
from pesq import pesq
import numpy as np
from moc import compare
from moc2 import compare as compare2
#from warpq import compute_WAPRQ as warpq
from lace_loss_metric import compare as laceloss_compare


parser = argparse.ArgumentParser()
parser.add_argument('folder', type=str, help='folder with processed items')
parser.add_argument('metric', type=str, choices=['pesq', 'moc', 'moc2', 'laceloss'], help='metric to be used for evaluation')


def get_bitrates(folder):
    with open(os.path.join(folder, 'bitrates.txt')) as f:
        x = f.read()

    bitrates = [int(y) for y in x.rstrip('\n').split()]

    return bitrates

def get_itemlist(folder):
    with open(os.path.join(folder, 'items.txt')) as f:
        lines = f.readlines()

    items = [x.split()[0] for x in lines]

    return items


def process_item(folder, item, bitrate, metric):
    fs, x_clean  = wavfile.read(os.path.join(folder, 'clean', f"{item}_{bitrate}_clean.wav"))
    fs, x_opus   = wavfile.read(os.path.join(folder, 'opus', f"{item}_{bitrate}_opus.wav"))
    fs, x_lace   = wavfile.read(os.path.join(folder, 'lace', f"{item}_{bitrate}_lace.wav"))
    fs, x_nolace = wavfile.read(os.path.join(folder, 'nolace', f"{item}_{bitrate}_nolace.wav"))

    x_clean  = x_clean.astype(np.float32) / 2**15
    x_opus   = x_opus.astype(np.float32) / 2**15
    x_lace   = x_lace.astype(np.float32) / 2**15
    x_nolace = x_nolace.astype(np.float32) / 2**15

    if metric == 'pesq':
        result = [pesq(fs, x_clean, x_opus), pesq(fs, x_clean, x_lace), pesq(fs, x_clean, x_nolace)]
    elif metric =='moc':
        result = [compare(x_clean, x_opus), compare(x_clean, x_lace), compare(x_clean, x_nolace)]
    elif metric =='moc2':
        result = [compare2(x_clean, x_opus), compare2(x_clean, x_lace), compare2(x_clean, x_nolace)]
    # elif metric == 'warpq':
        # result = [warpq(x_clean, x_opus), warpq(x_clean, x_lace), warpq(x_clean, x_nolace)]
    elif metric == 'laceloss':
        result = [laceloss_compare(x_clean, x_opus), laceloss_compare(x_clean, x_lace), laceloss_compare(x_clean, x_nolace)]
    else:
        raise ValueError(f'unknown metric {metric}')

    return result

def process_bitrate(folder, items, bitrate, metric):
    results = np.zeros((len(items), 3))

    for i, item in enumerate(items):
        results[i, :] = np.array(process_item(folder, item, bitrate, metric))

    return results


if __name__ == "__main__":
    args = parser.parse_args()

    items = get_itemlist(args.folder)
    bitrates = get_bitrates(args.folder)

    results = dict()
    for br in bitrates:
        print(f"processing bitrate {br}...")
        results[br] = process_bitrate(args.folder, items, br, args.metric)

    np.save(os.path.join(args.folder, f'results_{args.metric}.npy'), results)

    print("Done.")
