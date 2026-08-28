"""
This script is creates testvectors for qualifying a SILK enhancement method for use with the Opus codec according
to the internet draft https://datatracker.ietf.org/doc/draft-buethe-opus-speech-coding-enhancement/.
"""


import os
import argparse
import glob
from itertools import product
from functools import partial
from multiprocessing import Pool
import subprocess
import shutil

import yaml

DEFAULT_LICENSE = \
"""
Speech and music clips and their derived bitrates are provided under their original license.

Speech clips originate from the Mozilla CommonVoice dataset are provided under CC0 license (https://creativecommons.org/publicdomain/zero/1.0/)

Music clips (sample_N) have been extracted from clips retrieved from the Free Music Archive (https://freemusicarchive.org) and are provided
under a mix of CC BY (https://creativecommons.org/licenses/by/4.0/) and CC BY-SA (https://creativecommons.org/licenses/by-sa/3.0/) licenses. The
exact attributions and licenses are:

sample_1  : Track XXXVIII, A. Cooper, Free Music Archive, CC BY
sample_2  : Boogie!, AvapXia, Free Music Archive, CC BY
sample_3  : Chaotic Heart, Dirk Dehler, Free Music Archive, CC BY
sample_4  : Grass on the Field, Gagmesharkoff, Free Music Archive, CC BY
sample_5  : Douglas, Mephistopheles, Illinois Brass Band, Free Music Archive, CC BY
sample_6  : Spring Mvt 1 Allegro, John Harris with the Wichita State University Chamber Players, Free Music Archive, CC BY-SA
sample_7  : Spring Mvt 2 Largo, John Harris with the Wichita State University Chamber Players, Free Music Archive, CC BY-SA
sample_8  : Camille Saint-Saëns_Danse Macabre - Finale, Kevin MacLeod, Free Music Archive, CC BY
sample_9  : Spores (Stamets mix), Mindseye, Free Music Archive, CC BY
sample_10 : A Serpent I Did Hear, Pierce Murphy, Free Music Archive, CC BY
sample_11 : Self-Indulgent Jazz Interlude, ROZKOL, Free Music Archive, CC BY
sample_12 : flower lane, snoozy beats, Free Music Archive, CC BY
sample_13 : Ng'yahamba (ft. DJ Citie), Stella Jacobs, Free Music Archive, CC BY

Fullband speech clips (pNNN_MMM) are from the CSTR VCTK Corpus, The Centre for Speech Technology Research, University of Edinburgh
(Junichi Yamagishi, Christophe Veaux and Kirsten MacDonald). The audio recordings are provided under the Creative Commons
Attribution 4.0 International Public License (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/).

"""

DEBUG=False

parser = argparse.ArgumentParser()

parser.add_argument('clipdir', type=str, nargs='?', default=None, help='Input folder with test items (wideband/lowband clips + clips.yml); not required with --highband_only')
parser.add_argument('outputdir', type=str, help='Output folder')
parser.add_argument('--opus_demo', type=str, default='./opus_demo', help='reference opus_demo binary for generating bitstreams and reference output')
parser.add_argument('--highband_clips', type=str, default=None, help='folder with 48 kHz fullband clips (e.g. VCTK). If given, also generate highband (bandwidth-extension) testvectors under <outputdir>/highband')
parser.add_argument('--highband_only', action='store_true', help='only generate the highband testvectors (skip the lowband/WB generation); requires --highband_clips')
parser.add_argument('--verbose', type=int, default=0, help='verbosity level: 0 for quiet (default), 1 test-level logging, 2 for item-level logging')
parser.add_argument('--num_workers', type=int, default=10, help='pool size for multiprocessing (default: 10)')
parser.add_argument('--license', default=None, help="custom license file")

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


def sox(*call_args):
    try:
        call_args = ["sox"] + list(call_args)
        subprocess.run(call_args)
        return 0
    except:
        return 1

def create_test_bitstreams(out_dir : str,
                           clip_base_dir : str,
                           clip_dict : dict,
                           opus_demo : str,
                           bitrate,
                           bitrate_mode : str,
                           frame_length : int,
                           bandwidth : str,
                           encoder_complexity : int,
                           coding_mode_force_switch : bool=False,
                           verbose=0):

    assert bitrate_mode in {'cbr', 'vbr'} and "invalid bitrate_mode"
    assert encoder_complexity >= 0 and encoder_complexity <= 10 and isinstance(encoder_complexity, int) and "invalid encoder complexity"
    if isinstance(bitrate, int):
        assert (bitrate >= 6000 and bitrate <= 200000 and isinstance(bitrate, int)) and "invalid bitrate"
    else:
        assert isinstance(bitrate, str) and bitrate == "switching" and "invalid bitrate"
    assert bandwidth in {'WB'} and "invalid bandwidth (currently only WB supported)"
    assert os.path.isfile(opus_demo) and f"{opus_demo} is not a file"
    assert os.path.isdir(clip_base_dir) and f"{clip_base_dir} is not a directory"

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    testname = f"osce_test_{bitrate}_{bitrate_mode}_{frame_length}ms_{bandwidth}_c{encoder_complexity}" + ("_celtswitch" if coding_mode_force_switch else "_native")

    if verbose > 0:
        print(f"creating bitstreams for {testname}")

    test_dir = os.path.join(out_dir, testname)
    if os.path.exists(test_dir):
        print(f"warning: test directory {test_dir} exists")
    else:
        os.makedirs(test_dir)


    enc_options = ["-complexity", str(encoder_complexity), "-bandwidth", bandwidth, "-framesize", str(frame_length)]
    if bitrate_mode == "cbr": enc_options += ["-cbr"]
    if bitrate == "switching":
        enc_options += ["-osce_test_bitrate_switching"]
        br = 9000 # dummy bitrate
    else:
        br = bitrate
    if coding_mode_force_switch: enc_options += ["-osce_test_mode_switching"]

    # create bitstreams
    for group, clips in clip_dict.items():
        group_dir = os.path.join(test_dir, group)
        os.makedirs(group_dir, exist_ok=True)

        for clip in clips:
            clipname = os.path.splitext(os.path.basename(clip))[0]
            raw_clip = os.path.join(group_dir, clipname + ".raw")
            bitstream_path = os.path.join(group_dir, clipname + f"_{testname}.opus")
            if sox(os.path.join(clip_base_dir, clip), raw_clip): return 1
            if run_opus_encoder(opus_demo, raw_clip, bitstream_path, 'voip', 16000, 1, br, options=enc_options): return 1
            os.remove(raw_clip)

    return 0


def create_highband_bitstreams(hb_out_dir : str,
                               clip_paths : list,
                               opus_demo : str,
                               bitrate : int,
                               frame_length : int,
                               encoder_complexity : int=10,
                               verbose=0):
    """Generate highband (bandwidth-extension) testvectors.

    Each 48 kHz fullband clip is encoded as wideband (WB) at the given bitrate and
    frame length. The decoder-under-test later decodes these at 48 kHz with BWE and
    the highband is scored against a lowpass anchor derived from the reference, so no
    reference scores need to be baked here.
    """
    assert bitrate >= 6000 and bitrate <= 200000 and "invalid bitrate"
    assert os.path.isfile(opus_demo) and f"{opus_demo} is not a file"

    testname = f"osce_hbtest_{bitrate}_vbr_{frame_length}ms_WB_c{encoder_complexity}_native"

    if verbose > 0:
        print(f"creating highband bitstreams for {testname}")

    test_dir = os.path.join(hb_out_dir, testname)
    os.makedirs(test_dir, exist_ok=True)

    enc_options = ["-complexity", str(encoder_complexity), "-bandwidth", "WB", "-framesize", str(frame_length)]

    for clip in clip_paths:
        clipname = os.path.splitext(os.path.basename(clip))[0]
        raw_clip = os.path.join(test_dir, clipname + ".raw")
        bitstream_path = os.path.join(test_dir, f"{clipname}_{testname}.opus")
        if sox(clip, "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer", raw_clip): return 1
        if run_opus_encoder(opus_demo, raw_clip, bitstream_path, 'voip', 48000, 1, bitrate, options=enc_options): return 1
        os.remove(raw_clip)

    return 0


def generate_highband(output_dir, highband_clips, opus_demo, verbose=0, num_workers=10):
    hb_bitrates = [6000, 9000, 12000, 15000, 18000, 24000]
    hb_frame_lengths = [10, 20]
    hb_complexity = 10

    clip_paths = sorted(glob.glob(os.path.join(highband_clips, "*.wav")))
    if len(clip_paths) == 0:
        print(f"no wav clips found in {highband_clips}")
        return 1

    hb_out = os.path.join(output_dir, "highband")
    os.makedirs(hb_out, exist_ok=True)

    hb_args = [(hb_out, clip_paths, opus_demo, br, fl, hb_complexity)
               for br, fl in product(hb_bitrates, hb_frame_lengths)]
    p = Pool(num_workers)
    rvals = p.starmap(partial(create_highband_bitstreams, verbose=verbose), hb_args)
    if sum(rvals) > 0:
        print(f"failed to create {sum(rvals)} highband tests")
        return 1
    print("all highband tests created successfully!")

    # copy 48 kHz reference clips (used to derive the lowpass anchor at test time)
    hb_ref_dir = os.path.join(hb_out, "reference_clips")
    os.makedirs(hb_ref_dir, exist_ok=True)
    for clip in clip_paths:
        clipname = os.path.splitext(os.path.basename(clip))[0]
        sox(clip, "-c", "1", "-r", "48000", "-b", "16", "-e", "signed-integer",
            os.path.join(hb_ref_dir, f"{clipname}.s16"))

    return 0


def main(clip_dir, output_dir, opus_demo, verbose=0, num_workers=10, license=None, highband_clips=None, highband_only=False):

    bitrates = [6000, 9000, 12000, 15000, 18000, 24000, 32000, 'switching']
    bitrate_modes = ['cbr', 'vbr']
    frame_lengths = [10, 20]
    enc_complexities = [0, 2, 4, 6, 8, 10]
    force_switch=[False, True]

    if highband_only and highband_clips is None:
        raise SystemExit("error: --highband_only requires --highband_clips")
    if not highband_only and clip_dir is None:
        raise SystemExit("error: clipdir is required unless --highband_only is set")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if license is None:
        with open(os.path.join(output_dir, "LICENSE.txt"), "w") as f:
            f.write(DEFAULT_LICENSE)
    else:
        shutil.copy(license, os.path.join(output_dir, "LICENSE.txt"))

    if not highband_only:
        # load clips list
        with open(os.path.join(clip_dir, 'clips.yml'), "r") as f:
            clips = yaml.safe_load(f)

            test_args = []
            for br, m, l, c, f in product(bitrates, bitrate_modes, frame_lengths, enc_complexities, force_switch):
                test_args.append((output_dir, clip_dir, clips, opus_demo, br, m, l, 'WB', c, f))

            p = Pool(num_workers)
            rvals = p.starmap(partial(create_test_bitstreams, verbose=verbose), test_args)

            failed = sum(rvals)
            if failed > 0:
                print(f"failed to create {failed} tests")
                return 1
            else:
                print("all tests created successfully!")

        # copy references
        reference_dir = os.path.join(output_dir, "reference_clips")
        os.makedirs(reference_dir, exist_ok=True)
        for wavclip in os.listdir(os.path.join(clip_dir, 'clips')):
            if not wavclip.endswith(".wav"): continue
            clipname = wavclip[:-4]
            sox(os.path.join(clip_dir, 'clips', wavclip), os.path.join(reference_dir, f"{clipname}.s16"))

    if highband_clips is not None:
        if generate_highband(output_dir, highband_clips, opus_demo, verbose=verbose, num_workers=num_workers):
            return 1



if __name__ == "__main__":
    args = parser.parse_args()

    main(args.clipdir,
         args.outputdir,
         args.opus_demo,
         args.verbose,
         args.num_workers,
         args.license,
         args.highband_clips,
         args.highband_only
    )
