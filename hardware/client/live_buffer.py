"""Rolling time-window buffer for the live IMU dashboard (live_dashboard.py).

No socket or UI dependency, so it's testable on its own -- see
live_dashboard.py for how it's fed from imu_client.IMUStream/iter_rows, and
docs/superpowers/specs/2026-08-14-live-imu-dashboard-design.md for the design.
"""
from collections import deque

CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


class LiveBuffer:
    """Keeps only the last `window_seconds` of samples, evicted by the
    device's own t_us clock (not wall-clock arrival time, so a brief stall in
    delivery doesn't distort the window)."""

    def __init__(self, window_seconds=5.0):
        self.window_seconds = window_seconds
        self._rows = deque()  # each entry: (t_us, values) with values a 6-tuple

    def add_row(self, seq, t_us, values):
        self._rows.append((t_us, values))
        self._evict(t_us)

    def _evict(self, latest_t_us):
        window_us = self.window_seconds * 1_000_000
        while self._rows and (latest_t_us - self._rows[0][0]) > window_us:
            self._rows.popleft()

    def snapshot(self):
        """Returns {"t_s": [...], "ax": [...], ..., "gz": [...]} -- t_s is
        seconds relative to the oldest sample currently in the window, so a
        chart's x-axis stays a stable, increasing range. Empty lists if no
        data has arrived yet."""
        if not self._rows:
            return {"t_s": [], **{channel: [] for channel in CHANNELS}}

        t0 = self._rows[0][0]
        out = {"t_s": [(t_us - t0) / 1_000_000 for t_us, _ in self._rows]}
        for i, channel in enumerate(CHANNELS):
            out[channel] = [values[i] for _, values in self._rows]
        return out


def peak_g(buffer_snapshot):
    """Peak |accel| across a snapshot() from LiveBuffer -- the largest
    magnitude within the current window, not the whole session, so it
    reflects recent activity on a live view rather than a stale number."""
    rows = zip(buffer_snapshot["ax"], buffer_snapshot["ay"], buffer_snapshot["az"])
    magnitudes = [(ax * ax + ay * ay + az * az) ** 0.5 for ax, ay, az in rows]
    return max(magnitudes, default=0.0)
