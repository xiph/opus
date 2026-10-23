"""Regression checks for the informative OSCE qualification runner.

Run with: python3 scripts/osce/test_run_osce_tests.py
"""

import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import run_osce_tests as runner


class QualificationRunnerTests(unittest.TestCase):
    def test_comparison_tool_is_required(self):
        with patch('sys.stderr'):
            with self.assertRaises(SystemExit) as error:
                runner.parser.parse_args(['vectors', 'results'])
        self.assertEqual(error.exception.code, 2)

    def test_comparator_errors_propagate(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'broken_compare'
            executable.write_text('#!/bin/sh\nexit 9\n')
            executable.chmod(0o755)
            with self.assertRaises(subprocess.CalledProcessError) as error:
                runner.compute_moc_score('reference.s16', 'test.s16',
                                         osce_compare_bin=str(executable))
            self.assertEqual(error.exception.returncode, 9)

    def test_decoder_errors_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / 'broken_decoder'
            executable.write_text('#!/bin/sh\nexit 7\n')
            executable.chmod(0o755)
            self.assertEqual(runner.run_opus_decoder(str(executable), 'input.opus',
                                                    'output.s16', 16000, 1), 1)

    def run_lowband_with_scores(self, scores):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            testname = 'osce_test_12000_vbr_20ms_WB_c10_native'
            group = base / testname / 'en'
            group.mkdir(parents=True)
            for clip in range(len(scores)):
                (group / f'clip{clip}_{testname}.opus').touch()

            def decode(executable, bitstream, output, *args):
                Path(output).touch()
                return 0

            # Each pair is (test score, reference score), in decode order.
            with patch.object(runner, 'run_opus_decoder', side_effect=decode), \
                 patch.object(runner, 'compute_moc_score',
                              side_effect=[score for pair in scores for score in pair]):
                return runner.run_test(str(base), testname, str(base / 'results'),
                                       'test_decoder', [], ref_opus_demo='reference_decoder',
                                       ref_dec_options=[], osce_compare_bin='osce_compare')[0]

    def test_clip_threshold_equality_fails(self):
        # First clip has D = -0.5 exactly; the other clip makes the mean pass.
        self.assertFalse(self.run_lowband_with_scores([(0.05, 0.0), (0.0, 1.0)]))

    def test_clip_just_above_threshold_passes(self):
        self.assertTrue(self.run_lowband_with_scores(
            [(math.nextafter(0.05, 0.0), 0.0), (0.0, 1.0)]))

    def test_group_threshold_equality_fails(self):
        score = -runner.THRESHOLD_B * 0.1
        self.assertEqual(-score / 0.1, runner.THRESHOLD_B)
        self.assertFalse(self.run_lowband_with_scores([(score, 0.0)]))

    def test_group_just_above_threshold_passes(self):
        score = math.nextafter(-runner.THRESHOLD_B * 0.1, 0.0)
        self.assertTrue(self.run_lowband_with_scores([(score, 0.0)]))

    def test_empty_test_group_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            testname = 'osce_test_12000_vbr_20ms_WB_c10_native'
            (base / testname / 'en').mkdir(parents=True)
            passed, margin, reason = runner.run_test(
                str(base), testname, str(base / 'results'), 'test_decoder', [])
            self.assertFalse(passed)
            self.assertIsNone(margin)
            self.assertIn('no test bitstreams', reason)


if __name__ == '__main__':
    unittest.main()
