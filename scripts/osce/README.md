# Opus speech coding enhancement qualification tests

`run_osce_tests.py` runs the objective qualification tests described in
[draft-ietf-mlcodec-opus-speech-coding-enhancement](https://datatracker.ietf.org/doc/draft-ietf-mlcodec-opus-speech-coding-enhancement/).
The script is informative. The draft defines the pass criteria and the
`osce_compare` C tool defines the degradation and distortion metrics.
There is no Python metric backend.

## Requirements

- Python 3.9 or later, NumPy, and PyYAML:
  `python3 -m pip install -r scripts/osce/requirements.txt`
- The `osce_compare` executable. From an extracted Opus source distribution:
  `./configure --disable-doc` followed by `make osce_compare`.
- A decoder under test and, for lowband tests, an unenhanced conforming reference
  decoder. Both must accept the `opus_demo` command-line interface and produce
  headerless little-endian signed 16-bit mono PCM.
- The test vectors specified by the draft, extracted into `osce_testvectors_v1/`.
  The current location is
  <https://media.xiph.org/opus/ietf/osce_testvectors_v1.zip>.

For the reference implementation, speech enhancement is enabled at decoder
complexities 6 and above. The examples use `-dec_complexity 10` for the decoder
under test and `-dec_complexity 5` to disable enhancement in the reference
decoder. Configure Opus with `--enable-osce` to build an enhanced decoder.

Run the following commands from the Opus source directory. Paths to other
conforming decoder executables can be substituted for `./opus_demo`.

## Non-extending lowband test

```sh
python3 scripts/osce/run_osce_tests.py osce_testvectors_v1 results/lowband \
  --osce_compare ./osce_compare \
  --opus_demo ./opus_demo --opus_demo_options="-dec_complexity 10" \
  --reference_opus_demo ./opus_demo \
  --reference_opus_demo_options="-dec_complexity 5" \
  --decoder_delay 91
```

## Extending lowband test

```sh
python3 scripts/osce/run_osce_tests.py osce_testvectors_v1 results/extending \
  --osce_compare ./osce_compare --extending \
  --opus_demo ./opus_demo \
  --opus_demo_options="-dec_complexity 10 -enable_osce_bwe" \
  --reference_opus_demo ./opus_demo \
  --reference_opus_demo_options="-dec_complexity 5" \
  --extending_delay 273
```

## Highband test

```sh
python3 scripts/osce/run_osce_tests.py osce_testvectors_v1 results/highband \
  --osce_compare ./osce_compare --highband \
  --opus_demo ./opus_demo \
  --opus_demo_options="-dec_complexity 10 -enable_osce_bwe" \
  --extending_delay 273
```

The example delays (91 samples at 16 kHz and 273 at 48 kHz) apply to the
reference implementation's wideband decoding configuration. Determine the
actual delay for each decoder and sampling rate used. The current runner uses
the same delay for the reference and test decoders in each lowband test; use
appropriately aligned decoder wrappers if their delays differ.

Lowband tests write `test_results.txt`, with PASS/FAIL results, margins, and
a summary. Each test also writes `<TESTNAME>_moc.yml`, containing reference
and test scores by group and clip. Highband tests write `highband_results.txt`,
including PASS/FAIL or INFO for operating points below the draft's recommended
bitrate thresholds. Fullband clips belong in the vectors' `highband/` directory.

Run `python3 scripts/osce/run_osce_tests.py --help` for all options, including
worker count and verbosity. Always use the normative comparison tool and the
pass criteria from the applicable draft revision when reporting conformance.

Regression checks can be run with
`python3 scripts/osce/test_run_osce_tests.py`.
