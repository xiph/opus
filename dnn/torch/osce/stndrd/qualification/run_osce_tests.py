"""
This script is runs tests for qualifying a SILK enhancement method for use with the Opus codec according
to the internet draft https://datatracker.ietf.org/doc/draft-buethe-opus-speech-coding-enhancement/.
"""


import os
import argparse
from itertools import repeat
from functools import partial
from multiprocessing import Pool
import subprocess

import yaml
import numpy as np

from moc import compare as moc

DEBUG=False

class colors:
    PASS = '\033[94m'
    FAIL = '\033[91m'
    END = '\033[0m'

ALPHA=0.5

parser = argparse.ArgumentParser()

parser.add_argument('testdir', type=str, help='Input folder with test bitstreams and reference items')
parser.add_argument('outputdir', type=str, help='Output folder')
parser.add_argument('--opus_demo', type=str, default='./opus_demo', help='test opus_demo binary')
parser.add_argument('--opus_demo_options', type=str, default="", help='opus_demo option string (default="")')
parser.add_argument('--decoder_delay', type=int, default=91, help="decoder delay in samples @ 16kHz used for aligning decoded output to reference file (default: 91)")
parser.add_argument('--verbose', type=int, default=0, help='verbosity level: 0 for quiet (default), 1 test-level logging, 2 for group-level failure logging, >= 3 for full group-level logging')
parser.add_argument('--num_workers', type=int, default=10, help='pool size for multiprocessing (default: 10)')

def run_opus_encoder(opus_demo_path, input_pcm_path, bitstream_path, application, fs, num_channels, bitrate, options=[], verbose=False):

    call_args = [
        opus_demo_path,
        "-e",
        application,
        str(fs),
        str(num_channels),
        str(bitrate),
        "-bandwidth",
        "WB"
    ]

    call_args += options

    call_args += [
        input_pcm_path,
        bitstream_path
    ]

    try:
        if verbose:
            print(f"running {call_args}...")
            subprocess.run(call_args)
        else:
            subprocess.run(call_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        return 1

    return 0


def run_opus_decoder(opus_demo_path, bitstream_path, output_pcm_path, fs, num_channels, options=[], verbose=False):

    call_args = [
        opus_demo_path,
        "-d",
        str(fs),
        str(num_channels)
    ]

    call_args += options

    call_args += [
        bitstream_path,
        output_pcm_path
    ]

    try:
        if verbose:
            print(f"running {call_args}...")
            subprocess.run(call_args)
        else:
            subprocess.run(call_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        return 1

    return 0

def compute_moc_score(reference_pcm, test_pcm, delay=91):
    x_ref = np.fromfile(reference_pcm, dtype=np.int16).astype(np.float32) / (2 ** 15)
    x_cut = np.fromfile(test_pcm, dtype=np.int16).astype(np.float32) / (2 ** 15)

    moc_score = moc(x_ref, x_cut[delay:])

    return moc_score


def run_test(test_base_dir, testname, out_dir, opus_demo, dec_options, dec_delay=91, verbose=0):

    log_prefix = f"{testname}"

    if verbose > 0:
        print(f"{log_prefix:46s} starting test")

    ref_moc_file = os.path.join(test_base_dir, testname, f"reference_moc_scores_{testname}.yml")
    assert os.path.isfile(ref_moc_file) and f"missing reference moc scores for test {testname}"

    os.makedirs(out_dir, exist_ok=True)

    reference_dir = os.path.join(test_base_dir, 'reference_clips')

    with open(ref_moc_file, "r") as f:
        ref_mocs = yaml.load(f, yaml.FullLoader)

    mocs = dict()
    results = dict()
    passed = []
    min_rel_diff = 1000
    min_mean = 1000
    worst_clip = None
    worst_group = None

    for group in ref_mocs.keys():
        group_dir = os.path.join(test_base_dir, testname, group)
        results[group] = np.zeros((len(ref_mocs[group]), 2))
        mocs[group] = dict()
        clips = list(ref_mocs[group].keys())
        for i, (clipname, d_ref) in enumerate(ref_mocs[group].items()):
            bitstream = os.path.join(group_dir, f"{clipname}_{testname}.opus")
            dec_clip = os.path.join(out_dir, f"{clipname}_{testname}.dec.raw")
            ref_clip = os.path.join(reference_dir, f"{clipname}.s16")
            if run_opus_decoder(opus_demo, bitstream, dec_clip, 16000, 1, dec_options): return False
            d_test = compute_moc_score(ref_clip, dec_clip)
            results[group][i, 0] = d_ref
            results[group][i, 1] = d_test
            mocs[group][clipname] = [d_ref, d_test.item()]
            os.remove(dec_clip)


        rel_diff = ((results[group][:, 0] - results[group][:, 1]) /(0.1 + results[group][:, 0] ** ALPHA))

        min_idx = np.argmin(rel_diff).item()
        if rel_diff[min_idx] < min_rel_diff:
            min_rel_diff = rel_diff[min_idx]
            worst_clip = clips[min_idx]

        if np.mean(rel_diff) < min_mean:
            min_mean = np.mean(rel_diff).item()
            worst_group = group

        if np.min(rel_diff) < -0.5 or np.mean(rel_diff) < -0.052:
            if verbose > 1: print(f"{log_prefix:46s} FAIL {group} mean(rel_diff): {np.mean(rel_diff):5.4f} min(rel_diff): {np.min(rel_diff):5.4f} @ {clips[min_idx]}")
            passed.append(False)
        else:
            if verbose > 2: print(f"{log_prefix:46s} PASS {group} mean(rel_diff): {np.mean(rel_diff):5.4f} min(rel_diff): {np.min(rel_diff):5.4f} @ {clips[min_idx]}")
            passed.append(True)


    # save test results
    with open(os.path.join(out_dir, testname + "_moc.yml"), "w") as f:
        yaml.dump(mocs, f)

    if verbose > 2:
        print(f"{log_prefix:46s} worst group: {worst_group} ({min_mean})")
        print(f"{log_prefix:46s} worst clip:  {worst_clip} ({min_rel_diff})")

    return all(passed)



def main(test_dir, output_dir, opus_demo, decoder_options, verbose=0, num_workers=10):

    tests = sorted([x for x in os.listdir(test_dir) if x.startswith("osce_test_")])

    if len(decoder_options) > 0:
        dec_options = decoder_options.split(" ")
    else:
        dec_options = []

    print(f"found {len(tests)} tests")

    p = Pool(num_workers)
    results = p.starmap(partial(run_test, dec_delay=91, verbose=verbose), zip(repeat(test_dir), tests, repeat(output_dir), repeat(opus_demo), repeat(dec_options)))

    passed = 0
    with open(os.path.join(output_dir, 'test_results.txt'), "w") as f:
        for test, result in zip(tests, results):
            if result:
                f.write(f"{test:46s} PASS\n")
                if verbose:
                    print(f"{test:46s} " + colors.PASS + "PASS" + colors.END)
                passed += 1
            else:
                f.write(f"{test:46s} FAIL\n")
                if verbose:
                    print(f"{test:46s} " + colors.FAIL + "FAIL" + colors.END)

        if passed == len(results):
            f.write(f"all tests passed\n")
            print(f"all tests passed")
        else:
            f.write(f"{len(results) - passed} of {len(results)} tests failed\n")
            print(f"{len(results) - passed} of {len(results)} tests failed")




if __name__ == "__main__":
    args = parser.parse_args()

    main(args.testdir,
         args.outputdir,
         args.opus_demo,
         args.opus_demo_options,
         args.verbose,
         args.num_workers
    )
