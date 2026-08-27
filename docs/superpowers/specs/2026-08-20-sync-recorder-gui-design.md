# Sync Recorder GUI — Design

Date: 2026-08-20
Status: Approved for implementation

## Problem

`sync_recorder.py` (CLI) works but needs a filename decided upfront and typed on a command
line. The user wants: click Start, click Stop, *then* a native "Save As" dialog asks for a
name -- deciding the name after seeing whether the take was worth keeping, not before. A live
camera preview while recording ("real time tracker") is also wanted, as a framing/sanity check
during the take, not a live pose-estimation feature (RTMPose runs offline on saved clips
elsewhere in this project; running it live is a separate, much larger undertaking not
attempted here).

## Approach

**Tkinter**, not Streamlit: a native OS "Save As" dialog (`tkinter.filedialog.asksaveasfilename`)
is a desktop-app pattern Streamlit (browser-based) has no equivalent for. Tkinter is stdlib --
no new dependency beyond `opencv-python`, already added for `sync_recorder.py`.

**Record-to-temp, rename-on-save**: Start immediately begins writing to a temporary
prefix (so naming never blocks starting); Stop ends the recording and opens the save dialog;
the temp files are renamed to `<chosen name>_imu.csv` / `_video.mp4` / `_video_timestamps.csv`
/ `_alignment.json`. Canceling the dialog keeps the temp files under a timestamped name rather
than silently discarding a real take.

**Live preview**: the same webcam frame already being written to the video file is also
converted (`cv2.cvtColor` + `PIL.ImageTk`) and shown in a Tkinter `Label`, refreshed each
frame -- no second camera stream, just displaying what's already being captured.

**Reused, not rebuilt**: `capture_video()` and `record_imu()` from `sync_recorder.py` are
reused as-is for the actual capture loops (each already accepts a `stop_event`, matching a
Stop button naturally); only the orchestration (temp-file naming, the Tk window, the save
dialog) is new.

**Testable split**, same principle as `sync_recorder.py` and `live_dashboard.py`: the
temp-name generation and the rename-to-final-name logic are pure functions, unit-tested. The
Tk window/event loop itself is not unit-tested (no display/camera to drive it against in this
dev environment) -- manual QA on the user's machine, same as `live_dashboard.py`.

## Components

### `hardware/client/sync_recorder_gui.py` (new)

- `make_temp_prefix(base_dir, clock=time.time)` -- e.g. `<base_dir>/_recording_20260820_193045`,
  used the moment Start is clicked, before any name is known.
- `finalize_recording(temp_prefix, chosen_name)` -- renames the four temp output files
  (`_imu.csv`, `_video.mp4`, `_video_timestamps.csv`, `_alignment.json`) to
  `<chosen_name>_imu.csv` etc. Missing files (e.g. IMU never connected) are skipped, not an
  error -- a video-only take is still worth keeping.
- `RecorderApp` (Tk): Start button -> spawns the IMU + video capture threads (reusing
  `record_imu`/`capture_video` against a temp prefix) and starts the live preview loop;
  Stop button -> sets the shared `stop_event`, joins the threads, opens
  `filedialog.asksaveasfilename`, calls `finalize_recording`.

## Out of scope

- Live pose estimation / skeleton overlay during recording.
- Any change to `sync_recorder.py`'s CLI (kept as the non-GUI option) or `imu_client.py`.
