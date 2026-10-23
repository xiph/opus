This folder contains scripts for creating and running tests for qualifying a speech coding
enhancement method for use with Opus SILK.

The scripts have been tested with python 3.9, 3.10 and 3.11. Python dependencies are listed
in requirements.txt. In addition, the create_bistreams.py script will require sox to be
installed in PATH.

The objective test runner is now in scripts/osce/run_osce_tests.py at the repository
root. See scripts/osce/README.md for current instructions. It uses the normative
osce_compare C tool and does not require the Python metric implementation here.

Testvectors have been created from https://media.xiph.org/opus/ietf/osce_test_clips.zip using the script create_bitstreams.py.
