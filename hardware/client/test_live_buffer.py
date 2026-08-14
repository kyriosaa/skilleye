"""Tests for live_buffer.py. Standard library only:

    cd hardware/client && python -m unittest -v
"""

import socket
import threading
import unittest

from imu_client import IMUStream, StreamStats, iter_rows
from live_buffer import LiveBuffer, peak_g
from test_imu_client import DEVICE_HEADER, make_rows, serve_once


class TestLiveBuffer(unittest.TestCase):

    def test_snapshot_is_empty_before_any_data(self):
        buf = LiveBuffer(window_seconds=5.0)
        snap = buf.snapshot()
        self.assertEqual(snap["t_s"], [])
        for ch in ("ax", "ay", "az", "gx", "gy", "gz"):
            self.assertEqual(snap[ch], [])

    def test_snapshot_preserves_row_order_and_values(self):
        buf = LiveBuffer(window_seconds=5.0)
        buf.add_row(0, 0, (0.1, 0.2, 0.3, 1.0, 2.0, 3.0))
        buf.add_row(1, 500_000, (0.4, 0.5, 0.6, 4.0, 5.0, 6.0))

        snap = buf.snapshot()

        self.assertEqual(snap["t_s"], [0.0, 0.5])
        self.assertEqual(snap["ax"], [0.1, 0.4])
        self.assertEqual(snap["gz"], [3.0, 6.0])

    def test_evicts_samples_older_than_the_window(self):
        buf = LiveBuffer(window_seconds=2.0)  # 2_000_000 us
        for t_us, ax in [(0, 0.0), (1_000_000, 1.0), (2_000_000, 2.0), (3_000_000, 3.0)]:
            buf.add_row(0, t_us, (ax, 0, 0, 0, 0, 0))

        snap = buf.snapshot()

        # The t=0 sample is now 3s behind the latest (3s), outside the 2s
        # window; t_s is relative to the oldest sample *remaining*.
        self.assertEqual(snap["t_s"], [0.0, 1.0, 2.0])
        self.assertEqual(snap["ax"], [1.0, 2.0, 3.0])


class TestPeakG(unittest.TestCase):

    def test_empty_snapshot_has_zero_peak(self):
        buf = LiveBuffer()
        self.assertEqual(peak_g(buf.snapshot()), 0.0)

    def test_finds_the_largest_accel_magnitude_in_the_window(self):
        buf = LiveBuffer(window_seconds=5.0)
        buf.add_row(0, 0, (0.0, 0.0, 1.0, 0, 0, 0))       # |a| = 1.0 g
        buf.add_row(1, 100_000, (3.0, 4.0, 0.0, 0, 0, 0))  # |a| = 5.0 g -- the peak
        buf.add_row(2, 200_000, (0.0, 1.0, 0.0, 0, 0, 0))  # |a| = 1.0 g

        self.assertAlmostEqual(peak_g(buf.snapshot()), 5.0, places=6)


class TestLiveBufferOverSocket(unittest.TestCase):
    """Drives the real IMUStream/iter_rows against a mock TCP server that
    replays the firmware's exact wire format -- proves the full read path
    (socket -> parsing -> buffer) works without needing physical hardware."""

    def test_end_to_end_through_a_mock_device(self):
        payload = "\n".join(DEVICE_HEADER + make_rows(50)) + "\n"
        port, thread = serve_once(payload, chunk_size=11)

        buf = LiveBuffer(window_seconds=5.0)
        stats = StreamStats()
        with IMUStream("127.0.0.1", port, timeout=5.0) as stream:
            for seq, t_us, values in iter_rows(stream.lines(), stats):
                buf.add_row(seq, t_us, values)
        thread.join(timeout=5.0)

        snap = buf.snapshot()
        self.assertEqual(len(snap["t_s"]), 50)
        self.assertEqual(stats.rows, 50)
        self.assertEqual(stats.missing, 0)


if __name__ == "__main__":
    unittest.main()
