"""Tests for the pure (non-Tk) logic in sync_recorder_gui.py. The Tk window
itself isn't covered here -- no display/camera to drive it against in this
environment; see the design spec's "Testable split" section.

    cd hardware/client && python -m unittest -v
"""
import os
import tempfile
import unittest

from sync_recorder_gui import OUTPUT_SUFFIXES, finalize_recording, make_temp_prefix


def fake_clock(value):
    return lambda: value


class TestMakeTempPrefix(unittest.TestCase):

    def test_prefix_is_inside_the_given_base_dir(self):
        prefix = make_temp_prefix("/some/dir", clock=fake_clock(1_755_000_000))
        self.assertTrue(prefix.startswith("/some/dir" + os.sep) or
                         prefix.startswith("/some/dir/"))

    def test_two_calls_with_different_clock_values_differ(self):
        a = make_temp_prefix("/some/dir", clock=fake_clock(1_000))
        b = make_temp_prefix("/some/dir", clock=fake_clock(2_000))
        self.assertNotEqual(a, b)


class TestFinalizeRecording(unittest.TestCase):

    def test_renames_every_output_file_that_exists(self):
        with tempfile.TemporaryDirectory() as d:
            temp_prefix = os.path.join(d, "_recording_123")
            for suffix in OUTPUT_SUFFIXES:
                with open(temp_prefix + suffix, "w") as f:
                    f.write("x")

            chosen_name = os.path.join(d, "take01")
            finalize_recording(temp_prefix, chosen_name)

            for suffix in OUTPUT_SUFFIXES:
                self.assertTrue(os.path.exists(chosen_name + suffix), suffix)
                self.assertFalse(os.path.exists(temp_prefix + suffix), suffix)

    def test_skips_missing_files_without_raising(self):
        with tempfile.TemporaryDirectory() as d:
            temp_prefix = os.path.join(d, "_recording_123")
            # Only the video file exists -- e.g. the IMU never connected.
            with open(temp_prefix + "_video.mp4", "w") as f:
                f.write("x")

            chosen_name = os.path.join(d, "take01")
            finalize_recording(temp_prefix, chosen_name)  # must not raise

            self.assertTrue(os.path.exists(chosen_name + "_video.mp4"))
            self.assertFalse(os.path.exists(chosen_name + "_imu.csv"))

    def test_raises_nothing_when_no_files_exist_at_all(self):
        with tempfile.TemporaryDirectory() as d:
            temp_prefix = os.path.join(d, "_recording_nonexistent")
            chosen_name = os.path.join(d, "take01")
            finalize_recording(temp_prefix, chosen_name)  # must not raise


if __name__ == "__main__":
    unittest.main()
