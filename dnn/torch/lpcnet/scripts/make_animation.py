""" script for creating animations from debug data

"""


import argparse


import sys
sys.path.append('./')

from utils.endoscopy import make_animation, read_data

parser = argparse.ArgumentParser()

parser.add_argument('folder', type=str, help='endoscopy folder with debug output')
parser.add_argument('output', type=str, help='output file (will be auto-extended with .mp4)')

parser.add_argument('--start-index', type=int, help='index of first sample to be considered', default=0)
parser.add_argument('--stop-index', type=int, help='index of last sample to be considered', default=-1)
parser.add_argument('--interval', type=int, help='interval between frames in ms', default=20)
parser.add_argument('--half-window-length', type=int, help='half size of window for displaying signals', default=80)


if __name__ == "__main__":
    args = parser.parse_args()

    filename = args.output if args.output.endswith('.mp4') else args.output + '.mp4'
    data = read_data(args.folder)

    make_animation(
        data,
        filename,
        start_index=args.start_index,
        stop_index = args.stop_index,
        half_signal_window_length=args.half_window_length
    )
