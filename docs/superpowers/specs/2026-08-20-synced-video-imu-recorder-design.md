# Synchronized Video + IMU Recorder — Design

Date: 2026-08-20
Status: Approved for implementation

## Problem

`ml/skilleye/imu_fusion.py`'s `FusedBeginnerExpertModel` (README §2.7) still trains only on a
synthetic IMU signal derived from the skeleton -- real recordings exist for the sensor alone
(§2.10), but nothing syncs it with video, which is what the fusion model actually needs. The
user has a webcam attached to the same laptop that runs `imu_client.py`, so a single tool can
start both streams together ("one click") instead of manually launching two separate programs.

Scope note, stated plainly: this tool solves *recording*, not the harder part. A trustworthy
retrained fusion result still needs multiple people and multiple swings, not one clip -- this
spec is about making each individual recording session fast and well-synced, not about the
data-collection plan itself.

## Approach

### Two clocks, one shared anchor

The racket's IMU timestamps (`t_us`) are monotonic since the device's own boot -- no defined
relationship to the webcam's frame clock (existing `imu_client.py` docstring already says this).
Starting both "at the same time" from one script closes most of that gap (removes the human
delay of manually starting two separate programs) but doesn't make them the same clock. This
tool anchors both streams to the recording computer's wall clock (`time.time()`) at the moment
each one starts, and writes that anchor alongside the data, so a downstream step can convert
either stream's native timestamp into the shared wall-clock timeline. The existing tap-sync
convention (tap the racket once at the start) is kept as a physical backup/precision check --
starting together removes gross misalignment, the tap remains the fine-grained one.

### Testable without a camera or a physical racket

`hardware/client/live_dashboard.py` already established the pattern this follows: pure logic is
unit-tested, hardware-specific wiring is thin and untested (no camera or racket in this
environment to validate against). Concretely: the frame-capture loop takes an injected frame
source and video writer (duck-typed to `cv2.VideoCapture`/`cv2.VideoWriter`'s `.read()`/`.write()`
methods) rather than importing `cv2` itself -- `cv2` (not installed in this dev environment) is
only imported in the outermost CLI wiring, which real hardware would exercise. The IMU side
reuses `imu_client.py`'s already-tested `IMUStream`/`iter_rows` unchanged. The orchestration
(starting both together, writing the shared anchor) is integration-tested with the existing
mock-TCP-server pattern (`test_imu_client.py`'s `serve_once`/`DEVICE_HEADER`/`make_rows`) on the
IMU side and a fake frame source/writer on the video side.

## Components

### `hardware/client/sync_recorder.py` (new)

- `capture_video(frame_source, writer, timestamps_path, stop_event, max_frames=None)` -- pulls
  frames via `frame_source.read()` (returns `(ok, frame)`), calls `writer.write(frame)`, and
  appends `(frame_index, wall_clock_us)` to `timestamps_path` per frame. No `cv2` import; works
  with any object matching that interface. Stops on `stop_event.is_set()` or `max_frames`.
- `record_imu(imu_stream, out_path, stop_event)` -- thin wrapper around the existing
  `IMUStream`/`iter_rows` (imported from `imu_client.py`), writing the same CSV format
  `imu_client.py record` already produces, stopping on `stop_event`.
- `run_synced_recording(imu_stream_factory, frame_source_factory, writer_factory, out_prefix,
  seconds=None)` -- the orchestrator: creates both sides via the factories (real ones use
  `IMUStream(...)`/`cv2.VideoCapture(...)`/`cv2.VideoWriter(...)` in production, fakes in tests),
  starts both on a shared `threading.Event` so they begin together, records each side's
  wall-clock start, and writes `<out_prefix>_alignment.json` with both anchors plus metadata.
  Outputs: `<out_prefix>_imu.csv`, `<out_prefix>_video.mp4`, `<out_prefix>_video_timestamps.csv`,
  `<out_prefix>_alignment.json`.
- CLI (`main()`, mirrors `imu_client.py`'s `record` command): `python sync_recorder.py --out
  take01 --seconds 10 --camera 0`. Prints the same "tap the racket now" reminder
  `imu_client.py record` already prints.

## Out of scope

- Actually aligning/resampling the two streams into one training-ready array (a follow-up
  step, once real recordings exist -- `imu_fusion.py`'s synthetic-signal call site is already
  the documented swap-in point).
- A recording *plan* (how many people, how many swings) -- a data-collection/logistics
  question for the user's team, not a code design question.
- Any change to `imu_client.py` or `live_dashboard.py` -- both reused as-is.
