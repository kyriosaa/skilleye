"""
Desktop GUI over sync_recorder.py: click Start, click Stop, then a native
"Save As" dialog asks what to call the take -- naming happens after you've
seen whether it was worth keeping, not before. Also shows a live preview of
the webcam feed while recording (a framing/sanity check, not live pose
tracking -- this project's pose estimation runs offline on saved clips
elsewhere). Design:
docs/superpowers/specs/2026-08-20-sync-recorder-gui-design.md.

Run (from hardware/client/, needs `pip install opencv-python pillow`):

    python sync_recorder_gui.py

The pure logic below (temp-name generation, rename-on-save) is unit-tested
in test_sync_recorder_gui.py. The Tk window itself is not -- no display or
camera to drive it against in this dev environment; manual QA on your
machine, same as live_dashboard.py.
"""
import json
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from imu_client import DEFAULT_HOST, DEFAULT_PORT, IMUStream, StreamStats, load_config
from sync_recorder import capture_video, record_imu

OUTPUT_SUFFIXES = ["_imu.csv", "_video.mp4", "_video_timestamps.csv", "_alignment.json"]


def make_temp_prefix(base_dir, clock=time.time):
    """A throwaway output prefix used the moment Start is clicked, before any
    name is known -- e.g. <base_dir>/_recording_1755000000123456."""
    return os.path.join(base_dir, f"_recording_{int(clock() * 1_000_000)}")


def finalize_recording(temp_prefix, chosen_name):
    """Renames every OUTPUT_SUFFIXES file that exists from temp_prefix to
    chosen_name. Missing files (e.g. the IMU never connected, so there's no
    _imu.csv) are skipped, not an error -- a video-only take is still worth
    keeping."""
    for suffix in OUTPUT_SUFFIXES:
        src = temp_prefix + suffix
        if os.path.exists(src):
            os.replace(src, chosen_name + suffix)


# ------------------------------------------------------------------- GUI ----


class PreviewWriter:
    """Wraps a real cv2.VideoWriter so capture_video() (unchanged, no GUI
    awareness) also hands each frame to the preview -- write() is the only
    method capture_video() calls on its writer argument."""

    def __init__(self, real_writer, on_frame):
        self._real_writer = real_writer
        self._on_frame = on_frame

    def write(self, frame):
        self._real_writer.write(frame)
        self._on_frame(frame)


class RecorderApp:
    def __init__(self, root, host, port, camera_index, out_dir):
        self.root = root
        self.host = host
        self.port = port
        self.camera_index = camera_index
        self.out_dir = out_dir

        self.recording = False
        self.temp_prefix = None
        self.stop_event = None
        self.video_thread = None
        self.imu_thread = None
        self.cap = None
        self.real_writer = None
        self.imu_stats = None
        self.imu_error = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        root.title("SkillEye -- synced recorder")

        self.preview_label = tk.Label(root, text="(preview appears here once recording starts)")
        self.preview_label.pack(padx=8, pady=8)

        self.status_var = tk.StringVar(value="Idle")
        tk.Label(root, textvariable=self.status_var).pack(pady=(0, 8))

        button_frame = tk.Frame(root)
        button_frame.pack(pady=(0, 8))
        self.start_button = tk.Button(button_frame, text="Start", width=12, command=self.start)
        self.start_button.pack(side=tk.LEFT, padx=4)
        self.stop_button = tk.Button(button_frame, text="Stop", width=12,
                                      command=self.stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=4)

        self._preview_photo = None  # kept alive: Tk drops PhotoImages with no reference

    # -- start/stop --------------------------------------------------------

    def start(self):
        import cv2  # deferred -- only the GUI path touches real hardware

        self.temp_prefix = make_temp_prefix(self.out_dir)
        self.stop_event = threading.Event()
        self.imu_stats = None
        self.imu_error = None
        self.anchors = {}

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Camera error", f"Could not open camera {self.camera_index}.")
            return
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.real_writer = cv2.VideoWriter(f"{self.temp_prefix}_video.mp4", fourcc, 30.0,
                                            (width, height))
        writer = PreviewWriter(self.real_writer, self._set_latest_frame)

        self.video_thread = threading.Thread(
            target=capture_video,
            args=(self.cap, writer, f"{self.temp_prefix}_video_timestamps.csv", self.stop_event),
            daemon=True)
        self.imu_thread = threading.Thread(target=self._imu_worker, daemon=True)

        self.video_thread.start()
        self.imu_thread.start()
        # Anchors both streams to this computer's wall clock at the moment
        # each starts (same purpose as run_synced_recording's in
        # sync_recorder.py -- a bug earlier had this GUI path never write
        # this file at all, found from real recorded takes missing it).
        self.anchors["video_wall_clock_start_us"] = int(time.time() * 1_000_000)

        self.recording = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Recording -- tap the racket now to mark the sync point")
        self.root.after(33, self._update_preview)

    def _imu_worker(self):
        try:
            with IMUStream(self.host, self.port, timeout=10.0) as stream:
                self.anchors["imu_wall_clock_start_us"] = int(time.time() * 1_000_000)
                self.imu_stats = record_imu(stream, f"{self.temp_prefix}_imu.csv", self.stop_event)
        except (ConnectionRefusedError, OSError, TimeoutError) as exc:
            self.imu_error = str(exc)

    def _set_latest_frame(self, frame):
        with self.frame_lock:
            self.latest_frame = frame

    def _update_preview(self):
        if not self.recording:
            return
        with self.frame_lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
        if frame is not None:
            import cv2
            from PIL import Image, ImageTk
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._preview_photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
            self.preview_label.config(image=self._preview_photo, text="")
        self.root.after(33, self._update_preview)

    def stop(self):
        self.recording = False
        self.stop_event.set()
        self.video_thread.join(timeout=5.0)
        self.imu_thread.join(timeout=10.0)
        self.cap.release()
        self.real_writer.release()

        with open(f"{self.temp_prefix}_alignment.json", "w") as f:
            json.dump(self.anchors, f, indent=2)

        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("Stopped")

        if self.imu_error:
            messagebox.showwarning(
                "IMU not recorded",
                f"Video was recorded, but the IMU didn't connect: {self.imu_error}\n"
                "Check the racket's WiFi and try again if you need sensor data too.")

        chosen_path = filedialog.asksaveasfilename(
            initialdir=self.out_dir, title="Save recording as",
            defaultextension="", initialfile="take01")
        if chosen_path:
            # Strip an extension if the dialog added one (e.g. from a
            # filetype filter) -- OUTPUT_SUFFIXES supply their own.
            chosen_path = os.path.splitext(chosen_path)[0]
            finalize_recording(self.temp_prefix, chosen_path)
            self.status_var.set(f"Saved as {os.path.basename(chosen_path)}_*")
        else:
            self.status_var.set(f"Kept under temp name: {os.path.basename(self.temp_prefix)}_*")


def main():
    config = load_config()
    root = tk.Tk()
    app = RecorderApp(
        root, host=config.get("host", DEFAULT_HOST), port=config.get("port", DEFAULT_PORT),
        camera_index=0, out_dir=os.getcwd())
    root.mainloop()


if __name__ == "__main__":
    sys.exit(main())
