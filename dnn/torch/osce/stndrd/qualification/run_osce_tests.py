"""
This script is runs tests for qualifying a SILK enhancement method for use with the Opus codec according
to the internet draft https://datatracker.ietf.org/doc/draft-buethe-opus-speech-coding-enhancement/.
"""


import os
import argparse
import glob
from itertools import repeat
from functools import partial
from multiprocessing import Pool
import subprocess

import yaml
import numpy as np

from moc import compare as moc
from moc import highband_compare

DEBUG=False

class colors:
    PASS = '\033[94m'
    FAIL = '\033[91m'
    END = '\033[0m'

ALPHA=0.5

# lowband pass thresholds on the difference score D (see draft section 4.2.1):
#   per-clip:  D(CLIPNAME) > THRESHOLD_A for every clip in a group
#   per-group: mean of D over the group > THRESHOLD_B
THRESHOLD_A = -0.5
THRESHOLD_B = -0.052

parser = argparse.ArgumentParser()

parser.add_argument('testdir', type=str, help='Input folder with test bitstreams and reference items')
parser.add_argument('outputdir', type=str, help='Output folder')
parser.add_argument('--opus_demo', type=str, default='./opus_demo', help='test opus_demo binary')
parser.add_argument('--opus_demo_options', type=str, default="", help='opus_demo option string (default="")')
parser.add_argument('--reference_opus_demo', type=str, default=None, help='reference opus_demo binary (REQUIRED for the lowband test). REFMOC is computed by decoding each bitstream with this reference (unenhanced, conforming) decoder; any conforming Opus decoder is admissible. Not used by the highband test.')
parser.add_argument('--reference_opus_demo_options', type=str, default="-dec_complexity 5", help='decoder option string for --reference_opus_demo (default: "-dec_complexity 5", matching how the shipped reference scores were generated)')
parser.add_argument('--decoder_delay', type=int, default=91, help="decoder delay in samples @ 16kHz used for aligning decoded output to reference file (default: 91)")
parser.add_argument('--extending', action='store_true', help='evaluate a bandwidth-extending method: decode the test signal at 48 kHz and score the 0-8 kHz lowband against the 16 kHz reference. Enable the method itself via --opus_demo_options (e.g. -enable_osce_bwe).')
parser.add_argument('--extending_delay', type=int, default=None, help='decoder delay in samples @ 48kHz for aligning the extending (48 kHz) test output to the reference (default: 3 x --decoder_delay). The 48 kHz decode delay was measured to be ~3x the WB delay; the 16->48 kHz resampler adds no significant extra delay, and 3x keeps the lowband analysis frame-aligned with the 16 kHz reference scores.')
parser.add_argument('--highband', action='store_true', help='run the highband (bandwidth-extension) test instead of the lowband test: decode each <testdir>/highband bitstream at 48 kHz and check that the highband beats the lowpass anchor. Use a bandwidth-extending decoder via --opus_demo_options (e.g. -enable_osce_bwe).')
parser.add_argument('--highband_pass_rate', type=float, default=0.90, help='minimum per-bitrate clip pass rate required to pass a (gating) highband test (default: 0.90)')
parser.add_argument('--highband_threshold', type=float, default=0.0, help='highband pass margin tau: a clip passes if it beats the lowpass anchor by at least tau in every highband (default: 0.0)')
parser.add_argument('--highband_informal_max_bitrate', type=int, default=6000, help='highband tests at bitrates <= this value are reported as informal (INFO), not gating (default: 6000)')
parser.add_argument('--highband_informal_configs', type=str, default="9000:10", help='additional (bitrate:framesize_ms) configs to treat as informal (INFO, non-gating), comma-separated, e.g. "9000:10,12000:10". Default: "9000:10" (9 kb/s at 10 ms frames, where the WB baseband quality is too low to gate the bandwidth extension).')
parser.add_argument('--osce_compare', type=str, default=None, help='path to the osce_compare C binary. If given, the metric is computed by that binary instead of the (default) in-process Python moc implementation. Python remains the default.')
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

def _run_osce_compare(osce_compare_bin, args):
    out = subprocess.run([osce_compare_bin] + args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return out.stdout.strip()


def compute_moc_score(reference_pcm, test_pcm, fs_ref=16000, fs_test=16000, delay=91, osce_compare_bin=None):
    if osce_compare_bin is not None:
        out = _run_osce_compare(osce_compare_bin,
                                ["-fs_ref", str(fs_ref), "-fs_test", str(fs_test), "-delay", str(delay),
                                 reference_pcm, test_pcm])
        return float(out)
    x_ref = np.fromfile(reference_pcm, dtype=np.int16).astype(np.float32) / (2 ** 15)
    x_cut = np.fromfile(test_pcm, dtype=np.int16).astype(np.float32) / (2 ** 15)

    moc_score = moc(x_ref, x_cut[delay:], fs_x=fs_ref, fs_y=fs_test)

    return moc_score


def compute_highband_pass(reference_pcm, test_pcm, delay=273, threshold=0.0, osce_compare_bin=None):
    if osce_compare_bin is not None:
        out = _run_osce_compare(osce_compare_bin,
                                ["-highband", "-delay", str(delay), "-tau", str(threshold),
                                 reference_pcm, test_pcm]).split()
        margins = [float(t) for t in out[:4]]
        passed = out[-1] == "PASS"
        return passed, margins
    x_ref = np.fromfile(reference_pcm, dtype=np.int16).astype(np.float32) / (2 ** 15)
    x_test = np.fromfile(test_pcm, dtype=np.int16).astype(np.float32) / (2 ** 15)

    passed, margins = highband_compare(x_ref, x_test[delay:], fs=48000, threshold=threshold)

    return passed, margins


def run_highband_test(test_base_dir, testname, out_dir, opus_demo, dec_options, delay=273, threshold=0.0, osce_compare_bin=None, verbose=0):
    hb_dir = os.path.join(test_base_dir, 'highband')
    reference_dir = os.path.join(hb_dir, 'reference_clips')
    test_dir = os.path.join(hb_dir, testname)
    os.makedirs(out_dir, exist_ok=True)

    if verbose > 0:
        print(f"{testname:52s} starting highband test")

    bitstreams = sorted(glob.glob(os.path.join(test_dir, f"*_{testname}.opus")))
    n_pass = 0
    n_total = 0
    for bs in bitstreams:
        clipname = os.path.basename(bs)[:-len(f"_{testname}.opus")]
        dec_clip = os.path.join(out_dir, f"{clipname}_{testname}.hbdec.raw")
        ref_clip = os.path.join(reference_dir, f"{clipname}.s16")
        if run_opus_decoder(opus_demo, bs, dec_clip, 48000, 1, dec_options):
            return (testname, None, None)
        passed, _ = compute_highband_pass(ref_clip, dec_clip, delay=delay, threshold=threshold, osce_compare_bin=osce_compare_bin)
        os.remove(dec_clip)
        n_total += 1
        if passed:
            n_pass += 1
        elif verbose > 1:
            print(f"{testname:52s} FAIL clip {clipname}")

    return (testname, n_pass, n_total)


def run_test(test_base_dir, testname, out_dir, opus_demo, dec_options, ref_opus_demo=None, ref_dec_options=None, dec_delay=91, extending=False, extending_delay=273, osce_compare_bin=None, verbose=0):

    log_prefix = f"{testname}"

    # extending methods extend WB -> FB, so the test signal is decoded at 48 kHz and
    # scored on the 0-8 kHz lowband; the reference stays a 16 kHz (WB) decode.
    fs_test = 48000 if extending else 16000
    test_delay = extending_delay if extending else dec_delay

    if verbose > 0:
        print(f"{log_prefix:46s} starting test")

    os.makedirs(out_dir, exist_ok=True)

    reference_dir = os.path.join(test_base_dir, 'reference_clips')
    test_dir = os.path.join(test_base_dir, testname)

    mocs = dict()
    results = dict()
    passed = []
    min_rel_diff = 1000
    min_mean = 1000
    worst_clip = None
    worst_clip_group = None
    worst_group = None

    groups = sorted([g for g in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, g))])
    for group in groups:
        group_dir = os.path.join(test_dir, group)
        bitstreams = sorted(glob.glob(os.path.join(group_dir, f"*_{testname}.opus")))
        clips = [os.path.basename(bs)[:-len(f"_{testname}.opus")] for bs in bitstreams]
        results[group] = np.zeros((len(clips), 2))
        mocs[group] = dict()
        for i, clipname in enumerate(clips):
            bitstream = os.path.join(group_dir, f"{clipname}_{testname}.opus")
            dec_clip = os.path.join(out_dir, f"{clipname}_{testname}.dec.raw")
            ref_clip = os.path.join(reference_dir, f"{clipname}.s16")
            if run_opus_decoder(opus_demo, bitstream, dec_clip, fs_test, 1, dec_options): return (False, None, "decode error")
            d_test = compute_moc_score(ref_clip, dec_clip, fs_test=fs_test, delay=test_delay, osce_compare_bin=osce_compare_bin)
            os.remove(dec_clip)

            # REFMOC is always computed by decoding the same bitstream with the reference decoder,
            # at the same rate as the test signal (48 kHz in extending mode, with bandwidth extension
            # disabled, so that 16->48 kHz resampling effects on the lowband cancel between reference
            # and test).
            ref_dec_clip = os.path.join(out_dir, f"{clipname}_{testname}.refdec.raw")
            if run_opus_decoder(ref_opus_demo, bitstream, ref_dec_clip, fs_test, 1, ref_dec_options): return (False, None, "decode error")
            d_ref = compute_moc_score(ref_clip, ref_dec_clip, fs_test=fs_test, delay=test_delay, osce_compare_bin=osce_compare_bin)
            os.remove(ref_dec_clip)

            results[group][i, 0] = d_ref
            results[group][i, 1] = d_test
            mocs[group][clipname] = [float(d_ref), float(d_test)]


        rel_diff = ((results[group][:, 0] - results[group][:, 1]) /(0.1 + results[group][:, 0] ** ALPHA))

        min_idx = np.argmin(rel_diff).item()
        if rel_diff[min_idx] < min_rel_diff:
            min_rel_diff = rel_diff[min_idx]
            worst_clip = clips[min_idx]
            worst_clip_group = group

        if np.mean(rel_diff) < min_mean:
            min_mean = np.mean(rel_diff).item()
            worst_group = group

        if np.min(rel_diff) < THRESHOLD_A or np.mean(rel_diff) < THRESHOLD_B:
            if verbose > 1: print(f"{log_prefix:46s} FAIL {group} mean(rel_diff): {np.mean(rel_diff):5.4f} min(rel_diff): {np.min(rel_diff):5.4f} @ {clips[min_idx]}")
            passed.append(False)
        else:
            if verbose > 2: print(f"{log_prefix:46s} PASS {group} mean(rel_diff): {np.mean(rel_diff):5.4f} min(rel_diff): {np.min(rel_diff):5.4f} @ {clips[min_idx]}")
            passed.append(True)


    # save test results
    with open(os.path.join(out_dir, testname + "_moc.yml"), "w") as f:
        yaml.dump(mocs, f)

    # Margin to the fail thresholds (positive = headroom, negative = failing). The clip
    # criterion (worst single clip vs THRESHOLD_A) and the group-mean criterion (worst
    # group mean vs THRESHOLD_B) are both in difference-score units; the binding margin is
    # whichever is smaller, i.e. how close this test came to failing.
    clip_margin = min_rel_diff - THRESHOLD_A
    mean_margin = min_mean - THRESHOLD_B
    if clip_margin <= mean_margin:
        margin = clip_margin
        binding = f"clip {worst_clip_group}/{worst_clip}"
    else:
        margin = mean_margin
        binding = f"mean {worst_group}"

    if verbose > 2:
        print(f"{log_prefix:46s} worst group: {worst_group} ({min_mean})")
        print(f"{log_prefix:46s} worst clip:  {worst_clip} ({min_rel_diff})")
        print(f"{log_prefix:46s} margin to fail: {margin:+.4f} ({binding})")

    return (all(passed), float(margin), binding)



def main(test_dir, output_dir, opus_demo, decoder_options, reference_opus_demo=None, reference_decoder_options="-dec_complexity 5", decoder_delay=91, extending=False, extending_delay=None, osce_compare_bin=None, verbose=0, num_workers=10):

    tests = sorted([x for x in os.listdir(test_dir) if x.startswith("osce_test_")])

    if len(decoder_options) > 0:
        dec_options = decoder_options.split(" ")
    else:
        dec_options = []

    if reference_opus_demo is None:
        raise SystemExit("error: --reference_opus_demo is required; REFMOC is computed by decoding each bitstream with a reference (unenhanced, conforming) Opus decoder")
    assert os.path.isfile(reference_opus_demo), f"reference opus_demo binary {reference_opus_demo} not found"
    if len(reference_decoder_options) > 0:
        ref_dec_options = reference_decoder_options.split(" ")
    else:
        ref_dec_options = []
    print(f"computing reference scores with {reference_opus_demo} (options: '{reference_decoder_options}')")

    if extending and extending_delay is None:
        extending_delay = 3 * decoder_delay
    if extending:
        print(f"extending mode: decoding test signals at 48 kHz and scoring the lowband (extending_delay={extending_delay})")

    if osce_compare_bin is not None:
        assert os.path.isfile(osce_compare_bin), f"osce_compare binary {osce_compare_bin} not found"
        print(f"metric backend: osce_compare C binary ({osce_compare_bin})")

    print(f"found {len(tests)} tests")

    p = Pool(num_workers)
    results = p.starmap(partial(run_test, ref_opus_demo=reference_opus_demo, ref_dec_options=ref_dec_options, dec_delay=decoder_delay, extending=extending, extending_delay=extending_delay, osce_compare_bin=osce_compare_bin, verbose=verbose), zip(repeat(test_dir), tests, repeat(output_dir), repeat(opus_demo), repeat(dec_options)))

    passed = 0
    overall_min_margin = None
    overall_min_test = None
    with open(os.path.join(output_dir, 'test_results.txt'), "w") as f:
        for test, (ok, margin, binding) in zip(tests, results):
            status = "PASS" if ok else "FAIL"
            if margin is None:
                line = f"{test:46s} {status}  margin=n/a ({binding})"
            else:
                line = f"{test:46s} {status}  margin={margin:+.4f} ({binding})"
                if overall_min_margin is None or margin < overall_min_margin:
                    overall_min_margin = margin
                    overall_min_test = test
            f.write(line + "\n")
            if verbose:
                col = colors.PASS if ok else colors.FAIL
                print(col + line + colors.END)
            if ok:
                passed += 1

        if passed == len(results):
            summary = "all tests passed"
        else:
            summary = f"{len(results) - passed} of {len(results)} tests failed"
        if overall_min_margin is not None:
            summary += f"; tightest lowband margin {overall_min_margin:+.4f} @ {overall_min_test}"
        f.write(summary + "\n")
        print(summary)




def main_highband(test_dir, output_dir, opus_demo, decoder_options, decoder_delay=91, extending_delay=None, threshold=0.0, pass_rate=0.90, informal_max_bitrate=6000, informal_configs="9000:10", osce_compare_bin=None, verbose=0, num_workers=10):

    hb_dir = os.path.join(test_dir, 'highband')
    if not os.path.isdir(hb_dir):
        raise SystemExit(f"error: no highband/ subdirectory found in {test_dir}")

    tests = sorted([x for x in os.listdir(hb_dir) if x.startswith("osce_hbtest_")])
    if len(tests) == 0:
        raise SystemExit(f"error: no highband tests (osce_hbtest_*) found in {hb_dir}")

    informal_set = set()
    for pair in informal_configs.split(','):
        pair = pair.strip()
        if not pair:
            continue
        br, fs = pair.split(':')
        informal_set.add((int(br), int(fs)))

    dec_options = decoder_options.split(" ") if len(decoder_options) > 0 else []
    if "-enable_osce_bwe" not in dec_options:
        print("warning: --opus_demo_options does not contain -enable_osce_bwe; the highband test expects a bandwidth-extending decoder")

    delay = extending_delay if extending_delay is not None else 3 * decoder_delay

    if osce_compare_bin is not None:
        assert os.path.isfile(osce_compare_bin), f"osce_compare binary {osce_compare_bin} not found"
        print(f"metric backend: osce_compare C binary ({osce_compare_bin})")

    os.makedirs(output_dir, exist_ok=True)
    print(f"found {len(tests)} highband tests (delay={delay}, tau={threshold}, pass_rate>{pass_rate:.0%}, informal bitrates <= {informal_max_bitrate})")

    p = Pool(num_workers)
    results = p.starmap(partial(run_highband_test, delay=delay, threshold=threshold, osce_compare_bin=osce_compare_bin, verbose=verbose),
                        zip(repeat(test_dir), tests, repeat(output_dir), repeat(opus_demo), repeat(dec_options)))

    all_gating_pass = True
    with open(os.path.join(output_dir, 'highband_results.txt'), "w") as f:
        for testname, n_pass, n_total in results:
            if n_total is None or n_total == 0:
                line = f"{testname:52s} ERROR"
                all_gating_pass = False
            else:
                rate = n_pass / n_total
                parts = testname.split('_')
                bitrate = int(parts[2])
                framesize = int(parts[4].rstrip('ms'))
                informal = bitrate <= informal_max_bitrate or (bitrate, framesize) in informal_set
                if informal:
                    status = "INFO"
                else:
                    ok = rate > pass_rate
                    status = "PASS" if ok else "FAIL"
                    if not ok:
                        all_gating_pass = False
                line = f"{testname:52s} {status:5s} {n_pass}/{n_total} ({rate:.1%})"
            f.write(line + "\n")
            if verbose:
                print(line)

        summary = "all highband tests passed" if all_gating_pass else "highband test FAILED"
        f.write(summary + "\n")
        print(summary)



if __name__ == "__main__":
    args = parser.parse_args()

    if args.highband:
        main_highband(args.testdir,
                      args.outputdir,
                      args.opus_demo,
                      args.opus_demo_options,
                      args.decoder_delay,
                      args.extending_delay,
                      args.highband_threshold,
                      args.highband_pass_rate,
                      args.highband_informal_max_bitrate,
                      args.highband_informal_configs,
                      args.osce_compare,
                      args.verbose,
                      args.num_workers
        )
    else:
        main(args.testdir,
             args.outputdir,
             args.opus_demo,
             args.opus_demo_options,
             args.reference_opus_demo,
             args.reference_opus_demo_options,
             args.decoder_delay,
             args.extending,
             args.extending_delay,
             args.osce_compare,
             args.verbose,
             args.num_workers
        )
