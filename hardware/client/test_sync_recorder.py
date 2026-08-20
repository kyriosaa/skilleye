"""Tests for sync_recorder.py. Standard library only (plus the existing
test_imu_client.py mock-server helpers for the IMU side):

    cd hardware/client && python -m unittest -v
"""
import csv
import json
import os
import tempfile
import threading
import unittest

from sync_recorder import capture_video, record_imu, run_synced_recording
from imu_client import StreamStats
from test_imu_client import DEVICE_HEADER, make_rows, serve_once


class FakeFrameSource:
    """Stands in for cv2.VideoCapture -- same .read() -> (ok, frame) interface."""

    def __init__(self, frames):
        self._frames = list(frames)
        self._i = 0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        frame = self._frames[self._i]
        self._i += 1
        return True, frame


class FakeWriter:
    """Stands in for cv2.VideoWriter -- same .write() interface."""

    def __init__(self):
        self.written = []

    def write(self, frame):
        self.written.append(frame)


def fake_clock(sequence):
    """A clock() callable that returns each value in sequence in turn, in
    microseconds -- deterministic, no real timing dependency in tests."""
    it = iter(sequence)
    return lambda: next(it)


class TestCaptureVideo(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        # capture_video always writes a real timestamps file -- give every
        # test in this class a throwaway path in a directory that gets
        # cleaned up automatically, so tests that don't care about its
        # content (only writer.written) don't litter the working directory.
        self.timestamps_path = os.path.join(self._tmpdir.name, "timestamps.csv")

    def test_writes_every_frame_from_the_source_in_order(self):
        frames = ["frame0", "frame1", "frame2"]
        source = FakeFrameSource(frames)
        writer = FakeWriter()
        capture_video(source, writer, self.timestamps_path, threading.Event(),
                       clock=fake_clock([0, 1000, 2000]))
        self.assertEqual(writer.written, frames)

    def test_writes_timestamps_csv_with_frame_index_and_wall_clock(self):
        frames = ["a", "b"]
        source = FakeFrameSource(frames)
        writer = FakeWriter()
        capture_video(source, writer, self.timestamps_path, threading.Event(),
                       clock=fake_clock([5_000_000, 5_033_000]))
        with open(self.timestamps_path, newline="") as f:
            rows = list(csv.reader(f))
        self.assertEqual(rows[0], ["frame_index", "wall_clock_us"])
        self.assertEqual(rows[1], ["0", "5000000"])
        self.assertEqual(rows[2], ["1", "5033000"])

    def test_stops_when_the_source_is_exhausted(self):
        source = FakeFrameSource(["only_one"])
        writer = FakeWriter()
        capture_video(source, writer, self.timestamps_path, threading.Event(),
                       clock=fake_clock([0, 1, 2, 3, 4, 5]))
        self.assertEqual(len(writer.written), 1)

    def test_stops_when_the_stop_event_is_set(self):
        # Signal stop right after the 2nd frame is *written* (not mid-read,
        # which would be ambiguous about whether that in-flight frame counts)
        # -- capture_video checks stop_event at the top of its loop, so this
        # unambiguously means "exactly 2 frames, then stop before a 3rd".
        stop_event = threading.Event()

        class StoppingWriter(FakeWriter):
            def write(self, frame):
                super().write(frame)
                if len(self.written) == 2:
                    stop_event.set()

        source = FakeFrameSource(["a", "b", "c", "d", "e"])
        writer = StoppingWriter()
        capture_video(source, writer, self.timestamps_path, stop_event,
                       clock=fake_clock(range(10)))
        self.assertEqual(len(writer.written), 2)

    def test_respects_max_frames_even_if_more_are_available(self):
        source = FakeFrameSource(["a", "b", "c", "d"])
        writer = FakeWriter()
        capture_video(source, writer, self.timestamps_path, threading.Event(),
                       clock=fake_clock(range(10)), max_frames=2)
        self.assertEqual(len(writer.written), 2)


class TestRecordIMU(unittest.TestCase):

    def test_writes_the_same_csv_format_as_imu_client_record(self):
        payload = "\n".join(DEVICE_HEADER + make_rows(20)) + "\n"
        port, thread = serve_once(payload, chunk_size=9)

        from imu_client import IMUStream
        stats = StreamStats()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "imu.csv")
            with IMUStream("127.0.0.1", port, timeout=5.0) as stream:
                record_imu(stream, path, threading.Event(), stats=stats)
            thread.join(timeout=5.0)
            with open(path) as f:
                lines = f.read().splitlines()
        self.assertEqual(stats.rows, 20)
        self.assertTrue(any(line.startswith("seq,t_us") for line in lines))


class TestRunSyncedRecording(unittest.TestCase):

    def test_records_both_streams_with_a_shared_wall_clock_anchor(self):
        import time
        payload = "\n".join(DEVICE_HEADER + make_rows(30)) + "\n"
        port, thread = serve_once(payload, chunk_size=11)

        from imu_client import IMUStream

        class SlowFakeFrameSource(FakeFrameSource):
            """A tiny per-frame delay so the IMU side (real loopback socket
            I/O against the mock server) has enough wall-clock time to
            receive and process a few rows before capture_video finishes and
            triggers stop_event -- without this, video capture (all fake, no
            real I/O) can finish before the IMU thread gets scheduled at
            all, an artifact of faking one side and not the other."""
            def read(self):
                time.sleep(0.02)
                return super().read()

        frames = [f"frame{i}" for i in range(5)]

        with tempfile.TemporaryDirectory() as d:
            out_prefix = os.path.join(d, "take01")
            run_synced_recording(
                imu_stream_factory=lambda: IMUStream("127.0.0.1", port, timeout=5.0),
                frame_source_factory=lambda: SlowFakeFrameSource(frames),
                writer_factory=lambda frame_source: FakeWriter(),
                out_prefix=out_prefix,
            )
            thread.join(timeout=5.0)

            with open(f"{out_prefix}_alignment.json") as f:
                alignment = json.load(f)
            with open(f"{out_prefix}_imu.csv") as f:
                imu_lines = f.read().splitlines()
            with open(f"{out_prefix}_video_timestamps.csv", newline="") as f:
                video_rows = list(csv.reader(f))

        self.assertIn("imu_wall_clock_start_us", alignment)
        self.assertIn("video_wall_clock_start_us", alignment)
        self.assertTrue(any(line.startswith("0,") for line in imu_lines))
        self.assertEqual(len(video_rows) - 1, len(frames))  # minus the header row

    def test_a_broken_writer_factory_raises_instead_of_hanging(self):
        # Regression test for a real bug found on actual hardware: if the
        # video side fails to set up (there, two cv2.VideoCapture handles on
        # one camera conflicting) *before* go.set(), the IMU worker thread
        # was left blocked on go.wait() forever, and since it wasn't a
        # daemon thread, the whole process couldn't exit either. The test
        # itself runs run_synced_recording on a background thread with a
        # hard timeout, so if this regresses, the test fails cleanly
        # instead of hanging the whole suite.
        import tempfile, os
        payload = "\n".join(DEVICE_HEADER + make_rows(30)) + "\n"
        port, server_thread = serve_once(payload, chunk_size=11)

        from imu_client import IMUStream

        def broken_writer_factory(frame_source):
            raise RuntimeError("simulated camera conflict")

        result = {}

        def run():
            try:
                with tempfile.TemporaryDirectory() as d:
                    run_synced_recording(
                        imu_stream_factory=lambda: IMUStream("127.0.0.1", port, timeout=5.0),
                        frame_source_factory=lambda: FakeFrameSource(["a"]),
                        writer_factory=broken_writer_factory,
                        out_prefix=os.path.join(d, "take"),
                    )
            except Exception as e:
                result["error"] = e

        runner = threading.Thread(target=run, daemon=True)
        runner.start()
        runner.join(timeout=5.0)

        self.assertFalse(runner.is_alive(), "run_synced_recording hung instead of raising")
        self.assertIn("error", result)
        self.assertEqual(str(result["error"]), "simulated camera conflict")
        server_thread.join(timeout=5.0)


if __name__ == "__main__":
    unittest.main()
