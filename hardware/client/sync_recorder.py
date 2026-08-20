"""
Synchronized video + IMU recorder -- starts the webcam and the racket's IMU
stream together ("one click") instead of two separately-launched programs,
for building real (not synthetic) fusion-model training data. Design:
docs/superpowers/specs/2026-08-20-synced-video-imu-recorder-design.md.

The IMU's t_us and the webcam's frame clock are still two different clocks
(no hardware link between them) -- this anchors both to the recording
computer's wall clock at the moment each starts, written to
<out_prefix>_alignment.json, so a later step can line them up. Starting
together removes the human delay of launching two programs by hand; it does
not replace the tap-sync convention (tap the racket once at the start) as
the fine-grained sync point -- keep doing that too.

Run (from hardware/client/, needs `pip install opencv-python` first):

    python sync_recorder.py --out take01 --seconds 10
    python sync_recorder.py --out take01              # Ctrl-C to stop

Outputs: <out>_imu.csv, <out>_video.mp4, <out>_video_timestamps.csv,
<out>_alignment.json.
"""
import argparse
import csv
import json
import sys
import threading
import time

from imu_client import DEFAULT_HOST, DEFAULT_PORT, IMUStream, StreamStats, iter_rows, load_config


def wall_clock_us():
    """Default clock for this module: current wall-clock time in
    microseconds. Tests inject their own clock= (returning microseconds
    directly, no real timing dependency) instead of this."""
    return int(time.time() * 1_000_000)


# ---------------------------------------------------------------- video ----


def capture_video(frame_source, writer, timestamps_path, stop_event,
                   clock=wall_clock_us, max_frames=None):
    """Pulls frames from frame_source (an object with .read() -> (ok, frame),
    matching cv2.VideoCapture) and writes them to writer (matching
    cv2.VideoWriter's .write()) until the source is exhausted, stop_event is
    set, or max_frames is reached. Writes (frame_index, wall_clock_us) per
    frame to timestamps_path. No cv2 import here -- real hardware wiring
    lives in main(), this function works with any matching fakes."""
    frame_index = 0
    rows = []
    while not stop_event.is_set():
        if max_frames is not None and frame_index >= max_frames:
            break
        ok, frame = frame_source.read()
        if not ok:
            break
        writer.write(frame)
        rows.append((frame_index, int(clock())))
        frame_index += 1

    with open(timestamps_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_index", "wall_clock_us"])
        w.writerows(rows)


# ------------------------------------------------------------------ IMU ----


def record_imu(stream, out_path, stop_event, stats=None):
    """Records stream (an open IMUStream) to out_path in the same CSV format
    imu_client.py record already writes, stopping when stop_event is set or
    the device disconnects."""
    if stats is None:
        stats = StreamStats()
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        for line in stream.lines():
            if stop_event.is_set():
                break
            fh.write(line + "\n")
            for _ in iter_rows([line], stats):
                pass
    return stats


# ------------------------------------------------------------ orchestration --


def run_synced_recording(imu_stream_factory, frame_source_factory, writer_factory,
                          out_prefix, seconds=None, clock=wall_clock_us):
    """Starts IMU recording and video capture together (a shared
    threading.Event releases both at once), each anchored to the recording
    computer's wall clock at the moment it starts. Writes:
      <out_prefix>_imu.csv, <out_prefix>_video_timestamps.csv,
      <out_prefix>_alignment.json (the wall-clock anchors + metadata).
    frame_source_factory/writer_factory are called to build the video side
    (real callers pass cv2.VideoCapture/cv2.VideoWriter constructors; tests
    pass fakes) -- this function itself never imports cv2."""
    go = threading.Event()
    stop_event = threading.Event()
    anchors = {}
    imu_stats_holder = {}

    def imu_worker():
        with imu_stream_factory() as stream:
            go.wait()
            anchors["imu_wall_clock_start_us"] = int(clock())
            imu_stats_holder["stats"] = record_imu(stream, f"{out_prefix}_imu.csv", stop_event)

    thread = threading.Thread(target=imu_worker)
    thread.start()

    frame_source = frame_source_factory()
    writer = writer_factory()
    go.set()
    anchors["video_wall_clock_start_us"] = int(clock())

    if seconds is not None:
        timer = threading.Timer(seconds, stop_event.set)
        timer.start()
    capture_video(frame_source, writer, f"{out_prefix}_video_timestamps.csv",
                  stop_event, clock=clock)
    stop_event.set()  # make sure the IMU side also stops once video capture ends
    thread.join(timeout=10.0)

    with open(f"{out_prefix}_alignment.json", "w") as f:
        json.dump(anchors, f, indent=2)

    return anchors


# ------------------------------------------------------------------ CLI ----


def main(argv=None):
    config = load_config()
    parser = argparse.ArgumentParser(
        description="Record webcam video and the racket's IMU stream together.")
    parser.add_argument("--host", default=config.get("host", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=config.get("port", DEFAULT_PORT))
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    parser.add_argument("--out", required=True, help="output filename prefix")
    parser.add_argument("--seconds", type=float, default=None,
                         help="stop automatically after N seconds (else Ctrl-C)")
    parser.add_argument("--fps", type=float, default=30.0, help="webcam capture rate")
    args = parser.parse_args(argv)

    import cv2  # deferred: only the CLI path touches real hardware

    def frame_source_factory():
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"could not open camera index {args.camera}")
        return cap

    def writer_factory():
        cap = cv2.VideoCapture(args.camera)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        cap.release()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(f"{args.out}_video.mp4", fourcc, args.fps, (width, height))

    print("tap the racket now to mark the sync point, then start swinging.\n")
    print(f"connecting to {args.host}:{args.port} and camera {args.camera} ...")

    try:
        anchors = run_synced_recording(
            imu_stream_factory=lambda: IMUStream(args.host, args.port, timeout=10.0),
            frame_source_factory=frame_source_factory,
            writer_factory=writer_factory,
            out_prefix=args.out,
            seconds=args.seconds,
        )
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        print(f"\nconnection failed: {exc}", file=sys.stderr)
        print("check: laptop joined the device's WiFi network? device LED "
              "blinking (waiting) rather than off?", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped")
        return 0

    print(f"wrote {args.out}_imu.csv, {args.out}_video.mp4, "
          f"{args.out}_video_timestamps.csv, {args.out}_alignment.json")
    print(f"anchors: {anchors}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
