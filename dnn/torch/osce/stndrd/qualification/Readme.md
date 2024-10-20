This folder contains scripts for creating and running tests for qualifying a speech coding
enhancement method for use with Opus SILK.

The scripts have been tested with python 3.9, 3.10 and 3.11. Python dependencies are listed
in requirements.txt. In addition, the create_bistreams.py script will require sox to be
installed in PATH.

To run the test, download and extract testvectors https://media.xiph.org/opus/ietf/osce_testvectors_v0.zip.
The run_osce_tests.py script requires an executable of the enhanced opus decoder with
the same call signature as opus_demo, i.e.

EXECUTABLE -d SAMPLING_RATE NUMBER_OF_CHANNELS \[OPTIONS\] BITSTREAM PCM_OUTPUT

that produces a 16-bit PCM output file. To start the test, run

python run_osce_tests.py osce_testvectors_v0 my_test_output --opus-demo EXECUTABLE --opus-demo-options OPTIONS

which will prodce a text file my_test_output/test_results.txt containing the results (PASS/FAIL) for each
individual test. Furthermore, reference and test MOC scores are stored in yaml format in my_test_output/TESTNAME_moc.yml
under their group as primary key and clip name as secondary key.

Testvectors have been created from https://media.xiph.org/opus/ietf/osce_test_clips.zip using the script create_bitstreams.py.
