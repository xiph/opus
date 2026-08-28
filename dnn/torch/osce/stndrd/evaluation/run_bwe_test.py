"""
/* Copyright (c) 2023 Amazon
   Written by Jan Buethe */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

"""Blind bandwidth-extension evaluation.

For every 48 kHz test clip the clip is processed with opus_demo in full
(encode + decode) mode using blind bandwidth extension

    opus_demo <application> 48000 1 <bitrate> -bandwidth WB -dec_complexity 5 -enable_osce_bwe <in> <out>

and the decoded output is scored against the original clip with the highband
criterion from highband_eval.py: a clip passes if the reference-to-test
band-wise distortion does not exceed the reference-to-lowpass distortion in any
of the evaluated highbands. Results are aggregated over all clips and reported
per bitrate.
"""

import os
import sys
import csv
import glob
import functools
import argparse
import subprocess
import concurrent.futures

import numpy as np
from scipy.signal import stft

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from highband_eval import band_wise_distortion, opus_eband10ms, highband_idx

# STFT bin resolution: 48000 Hz / nperseg (480) = 100 Hz per bin.
_BIN_HZ = 48000 / 480

# Frequency edges (Hz) of the highbands that band_wise_distortion returns.
# It returns the last -highband_idx bands of the opus_eband10ms filter bank,
# so we need one extra leading edge (highband_idx - 1) to bracket them.
_highband_edges = opus_eband10ms[highband_idx - 1:]
_highband_ranges = [
    (_highband_edges[i] * _BIN_HZ, _highband_edges[i + 1] * _BIN_HZ)
    for i in range(len(_highband_edges) - 1)
]

parser = argparse.ArgumentParser()
parser.add_argument('opus_demo', type=str, help='path to opus_demo binary (must support -enable_osce_bwe)')
parser.add_argument('inputdir', type=str, help='input folder with 48 kHz test clips (wav)')
parser.add_argument('bitrates', type=str, help='comma-separated list of bitrates in bps, e.g. "9000,12000,15000"')
parser.add_argument('--application', type=str, default='voip', help='opus_demo application (default: voip)')
parser.add_argument('--dec_complexity', type=int, default=5, help='decoder complexity passed to opus_demo (default: 5; >=6 enables LACE, >=7 enables NoLACE lowband enhancement)')
parser.add_argument('--delay', type=int, default=0, help='samples to trim from the start of the decoded output to compensate the codec delay (48 kHz, default: 0)')
parser.add_argument('--workdir', type=str, default=None, help='directory for intermediate raw files (default: a temporary directory)')
parser.add_argument('--csv', type=str, default=None, help='write per-clip per-band distortions to this CSV file for comparison analysis')
parser.add_argument('--pass-margin', type=str, default='0', dest='pass_margin',
                    help='pass threshold on min-band (dist_ref_lp - dist_ref_test). '
                         'Either a single float applied to all bitrates (e.g. "0.02"), '
                         'or a per-bitrate schedule "9000:-0.221,12000:-0.003,15000:0.007" '
                         '(bitrates not listed use 0). Default: 0 (original rule).')
parser.add_argument('--jobs', type=int, default=os.cpu_count(), help='number of parallel worker processes (default: number of CPUs)')
parser.add_argument('--verbose', action='store_true', help='list the failing clips per bitrate')


def parse_pass_margin(spec):
    """Return (default_margin, {bitrate_str: margin}) from a --pass-margin spec."""
    spec = (spec or '0').strip()
    if ':' not in spec:
        return float(spec), {}
    schedule = {}
    for pair in spec.split(','):
        pair = pair.strip()
        if not pair:
            continue
        br, val = pair.split(':')
        schedule[br.strip()] = float(val)
    return 0.0, schedule


def sox(*call_args):
    subprocess.run(["sox"] + list(call_args), check=True)


def run_opus_bwe(opus_demo_path, application, bitrate, input_pcm_path, output_pcm_path, dec_complexity=5, verbose=False):
    call_args = [
        opus_demo_path,
        application,
        "48000",
        "1",
        str(bitrate),
        "-bandwidth", "WB",
        "-dec_complexity", str(dec_complexity),
        "-enable_osce_bwe",
        input_pcm_path,
        output_pcm_path
    ]

    if verbose:
        print(f"running {' '.join(call_args)}...")
        subprocess.run(call_args, check=True)
    else:
        subprocess.run(call_args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def highband_pass(x_ref, x_test, delay=0, pass_margin=0.0):
    """Return (passed, dist_ref_test, dist_ref_lp) following highband_eval.py.

    A clip passes when, in every evaluated highband, the reference-to-test
    distortion stays at least `pass_margin` below the reference-to-lowpass
    anchor, i.e. min_band(dist_ref_lp - dist_ref_test) >= pass_margin.
    pass_margin=0 reproduces the original "beat the anchor in all bands" rule;
    positive tightens it, negative relaxes it.
    """
    if delay > 0:
        x_test = x_test[delay:]

    n = min(len(x_ref), len(x_test))
    x_ref = x_ref[:n].astype(np.float64)
    x_test = x_test[:n].astype(np.float64)

    m = np.max(np.abs(x_ref))
    if m == 0:
        return False, None, None
    x_ref = x_ref / m
    x_test = x_test / m

    _, _, X_ref = stft(x_ref, nperseg=480)
    _, _, X_test = stft(x_test, nperseg=480)

    # lowpass anchor: reference with the highband removed (i.e. no extension)
    X_lp = X_ref.copy()
    X_lp[80:, :] = 0

    dist_ref_test = band_wise_distortion(X_ref, X_test, 2)
    dist_ref_lp = band_wise_distortion(X_ref, X_lp, 2)

    passed = bool(np.all((dist_ref_lp - dist_ref_test) >= pass_margin))

    return passed, dist_ref_test, dist_ref_lp


def _fmt_band(values):
    return "[" + ", ".join(f"{v:7.3f}" for v in values) + "]"


def _convert_clip(clip_path, processdir):
    """Convert one clip to 48 kHz mono 16-bit raw (bitrate-independent)."""
    clipname = os.path.splitext(os.path.basename(clip_path))[0]
    clean_path = os.path.join(processdir, clipname + "_clean.s16")
    sox(clip_path, "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer", clean_path)
    return clip_path, clean_path


def _eval_task(opus_demo, application, bitrate, clipname, clean_path, processdir,
               dec_complexity, delay, pass_margin, verbose):
    """Decode one (bitrate, clip) with BWE and score it. Returns a result tuple."""
    out_path = os.path.join(processdir, clipname + f"_{bitrate}_bwe.s16")
    try:
        run_opus_bwe(opus_demo, application, bitrate, clean_path, out_path,
                     dec_complexity=dec_complexity, verbose=verbose)
    except subprocess.CalledProcessError as e:
        return (bitrate, clipname, None, None, None, f"opus_demo failed (exit {e.returncode})")

    x_ref = np.fromfile(clean_path, dtype=np.int16).astype(np.float32)
    x_test = np.fromfile(out_path, dtype=np.int16).astype(np.float32)
    passed, dist_ref_test, dist_ref_lp = highband_pass(x_ref, x_test, delay=delay, pass_margin=pass_margin)
    return (bitrate, clipname, passed, dist_ref_test, dist_ref_lp, None)


def main(opus_demo, inputdir, bitrates, application, dec_complexity, delay, workdir, csv_path, pass_margin_spec, jobs, verbose):
    clips = sorted(glob.glob(os.path.join(inputdir, '*.wav')))
    if len(clips) == 0:
        print(f"no wav clips found in {inputdir}")
        return

    bitrate_list = [br.strip() for br in bitrates.split(',') if br.strip()]

    if workdir is None:
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        processdir = tmp.name
    else:
        tmp = None
        processdir = workdir
        os.makedirs(processdir, exist_ok=True)

    jobs = max(1, jobs or 1)
    default_margin, margin_schedule = parse_pass_margin(pass_margin_spec)
    margin_for = lambda br: margin_schedule.get(br, default_margin)
    band_labels = [f"{int(lo)}-{int(hi)}Hz" for lo, hi in _highband_ranges]
    print(f"evaluating {len(clips)} clips at bitrates {bitrate_list} "
          f"(application={application}, dec_complexity={dec_complexity}, delay={delay}, jobs={jobs})")
    print(f"highbands evaluated: {band_labels}")
    print(f"pass margins: {{" + ", ".join(f'{br}:{margin_for(br):+g}' for br in bitrate_list) + "}\n")

    # Phase 1: convert each clip once (bitrate-independent), in parallel.
    # Phase 2: decode+score every (bitrate, clip) task, in parallel.
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        clean_map = dict(ex.map(functools.partial(_convert_clip, processdir=processdir), clips))

        futures = []
        for bitrate in bitrate_list:
            for clip_path in clips:
                clipname = os.path.splitext(os.path.basename(clip_path))[0]
                futures.append(ex.submit(
                    _eval_task, opus_demo, application, bitrate, clipname,
                    clean_map[clip_path], processdir, dec_complexity, delay,
                    margin_for(bitrate), verbose))
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    # Group results by bitrate, keep clip order deterministic.
    by_bitrate = {br: {} for br in bitrate_list}
    for bitrate, clipname, passed, d_test, d_lp, err in results:
        by_bitrate[bitrate][clipname] = (passed, d_test, d_lp, err)

    csv_rows = []
    summary = dict()
    for bitrate in bitrate_list:
        num_passed = 0
        num_failed = 0
        num_errors = 0
        failed_clips = []

        for clip_path in clips:
            clipname = os.path.splitext(os.path.basename(clip_path))[0]
            passed, dist_ref_test, dist_ref_lp, err = by_bitrate[bitrate][clipname]

            if err is not None:
                num_errors += 1
                num_failed += 1
                failed_clips.append((clipname, None, None, err))
                continue

            if dist_ref_test is not None:
                for band_idx, (lo, hi) in enumerate(_highband_ranges):
                    d_test = float(dist_ref_test[band_idx])
                    d_lp = float(dist_ref_lp[band_idx])
                    csv_rows.append({
                        'bitrate': bitrate,
                        'clip': clipname,
                        'passed': int(passed),
                        'band': band_idx,
                        'freq_lo_hz': int(lo),
                        'freq_hi_hz': int(hi),
                        'dist_ref_test': d_test,
                        'dist_ref_lp': d_lp,
                        # positive margin => test beats the lowpass anchor in this band
                        'margin': d_lp - d_test,
                    })

            if passed:
                num_passed += 1
            else:
                num_failed += 1
                failed_clips.append((clipname, dist_ref_test, dist_ref_lp, None))

        total = num_passed + num_failed
        summary[bitrate] = (num_passed, total)
        errnote = f", {num_errors} errored" if num_errors else ""
        print(f"bitrate {bitrate}: {num_passed}/{total} passed, {num_failed} failed{errnote}")
        for clipname, dist_ref_test, dist_ref_lp, err in failed_clips:
            if err is not None:
                print(f"    FAIL {clipname}  ({err})")
            else:
                print(f"    FAIL {clipname}")
                if dist_ref_test is not None:
                    print(f"        ref-test: {_fmt_band(dist_ref_test)}")
                    print(f"        ref-lp  : {_fmt_band(dist_ref_lp)}")
                    print(f"        margin  : {_fmt_band(dist_ref_lp - dist_ref_test)}")

    print("\n=== summary ===")
    total_passed = 0
    total_count = 0
    for bitrate in bitrate_list:
        num_passed, total = summary[bitrate]
        total_passed += num_passed
        total_count += total
        print(f"bitrate {bitrate}: {num_passed}/{total} passed")
    print(f"overall: {total_passed}/{total_count} passed")

    if csv_path is not None and csv_rows:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nwrote per-band data for {len(csv_rows)} rows to {csv_path}")

    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.opus_demo,
         args.inputdir,
         args.bitrates,
         args.application,
         args.dec_complexity,
         args.delay,
         args.workdir,
         args.csv,
         args.pass_margin,
         args.jobs,
         args.verbose)
