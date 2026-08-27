"""
Live view of the racket-mounted IMU stream (hardware/firmware/firmware.ino)
while it's actually being swung -- a hardware sanity check ("does the sensor
respond the way a swing should look, right now") and a demo asset. Separate
from the validated skilleye-demo pipeline on purpose. Design:
docs/superpowers/specs/2026-08-14-live-imu-dashboard-design.md.

Run (from hardware/client/):

    streamlit run live_dashboard.py

Reuses imu_client.py's IMUStream/iter_rows/StreamStats as-is -- this file
only adds the background-thread wiring and the Streamlit rendering, neither
of which is unit-tested (see live_buffer.py / test_live_buffer.py for the
part that is). Manual QA against the physical racket happens on the user's
machine, not in this environment.
"""
import threading
import time

import pandas as pd
import streamlit as st

from imu_client import DEFAULT_HOST, DEFAULT_PORT, IMUStream, StreamStats, iter_rows, load_config
from live_buffer import LiveBuffer, peak_g

WINDOW_SECONDS = 5.0
POLL_SECONDS = 0.15
# Rendering runs in bursts of this many polls, then st.rerun()s -- a plain
# `while True` would never give Streamlit a chance to notice the Disconnect
# button was clicked, since widget state is only re-read between reruns.
BURST_POLLS = 20


class StreamWorker:
    """Owns the background thread that reads the device socket. Everything
    the render loop reads (buffer, stats, error, connected) is guarded by
    `lock`, since it's written from a different thread."""

    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.buffer = LiveBuffer(window_seconds=WINDOW_SECONDS)
        self.stats = StreamStats()
        self.lock = threading.Lock()
        self.error = None
        self.connected = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        try:
            with IMUStream(self.host, self.port, timeout=10.0) as stream:
                with self.lock:
                    self.connected = True
                for seq, t_us, values in iter_rows(stream.lines(), self.stats):
                    if self._stop.is_set():
                        break
                    with self.lock:
                        self.buffer.add_row(seq, t_us, values)
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            with self.lock:
                self.error = (
                    f"{exc} -- check: laptop joined the device's WiFi network? "
                    f"device LED blinking (waiting) rather than off?")
        finally:
            with self.lock:
                self.connected = False

    def snapshot(self):
        with self.lock:
            return {
                "buffer": self.buffer.snapshot(),
                "rows": self.stats.rows,
                "missing": self.stats.missing,
                "hz": self.stats.measured_hz,
                "error": self.error,
                "connected": self.connected,
            }


def render_snapshot(state, accel_chart, gyro_chart, metrics):
    buf = state["buffer"]
    if buf["t_s"]:
        df = pd.DataFrame({"t_s": buf["t_s"], "ax": buf["ax"], "ay": buf["ay"], "az": buf["az"]})
        accel_chart.line_chart(df, x="t_s", y=["ax", "ay", "az"], height=250)
        df = pd.DataFrame({"t_s": buf["t_s"], "gx": buf["gx"], "gy": buf["gy"], "gz": buf["gz"]})
        gyro_chart.line_chart(df, x="t_s", y=["gx", "gy", "gz"], height=250)

    cols = metrics.columns(4)
    cols[0].metric("Rows", state["rows"])
    cols[1].metric("Rate", f"{state['hz']:.0f} Hz")
    cols[2].metric("Dropped", state["missing"])
    cols[3].metric(f"Peak |a| (last {WINDOW_SECONDS:.0f}s)", f"{peak_g(buf):.2f} g")


def main():
    st.set_page_config(page_title="SkillEye -- live IMU", layout="wide")
    st.title("Live IMU stream")
    st.caption("Racket-mounted accelerometer/gyroscope, straight off the device -- "
               "a hardware sanity check, not the swing-quality demo.")

    config = load_config()
    if "worker" not in st.session_state:
        st.session_state.worker = None

    with st.sidebar:
        host = st.text_input("Host", value=config.get("host", DEFAULT_HOST))
        port = st.number_input("Port", value=int(config.get("port", DEFAULT_PORT)), step=1)
        connect_disabled = st.session_state.worker is not None
        if st.button("Connect", disabled=connect_disabled):
            worker = StreamWorker(host, int(port))
            worker.start()
            st.session_state.worker = worker
            st.rerun()
        if st.button("Disconnect", disabled=not connect_disabled):
            st.session_state.worker.stop()
            st.session_state.worker = None
            st.rerun()

    worker = st.session_state.worker
    if worker is None:
        st.info("Not connected. Set the host/port in the sidebar and click Connect.")
        return

    accel_chart = st.empty()
    gyro_chart = st.empty()
    metrics = st.empty()

    for _ in range(BURST_POLLS):
        state = worker.snapshot()
        if state["error"]:
            st.error(state["error"])
            st.session_state.worker = None
            return
        render_snapshot(state, accel_chart, gyro_chart, metrics)
        time.sleep(POLL_SECONDS)

    st.rerun()  # yield back to Streamlit so a Disconnect click gets noticed


if __name__ == "__main__":
    main()
