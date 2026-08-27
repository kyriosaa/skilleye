# Live IMU Dashboard — Design

Date: 2026-08-14
Status: Approved for implementation

## Problem

The racket-mounted IMU (`hardware/firmware/firmware.ino`, ESP32-C6 + GY-521) already streams
live CSV over its own WiFi AP, and `hardware/client/imu_client.py` already has a working
`monitor` command that prints a live rate/peak-g readout to the terminal. What's missing is a
visual, on-computer view of the stream while the racket is actually being swung — useful both
as a hardware sanity check (does the sensor respond the way a swing should look, right now,
not after the fact in a recorded CSV) and as a demo asset.

## Approach: thin Streamlit UI over the existing client, kept separate from the validated demo

Reuse `imu_client.py`'s `IMUStream` (TCP connection) and `iter_rows`/`StreamStats` (CSV
parsing, rate/drop tracking) as-is -- no reimplementing the wire protocol. Add:

1. A new, pure buffering class (`LiveBuffer`) that turns the incoming row stream into a
   rolling time window ready to plot -- this has no socket or UI dependency, so it's fully
   unit-testable.
2. A new, thin Streamlit script (`live_dashboard.py`) that runs a background thread reading
   the socket into a `LiveBuffer`, and a render loop that redraws a chart + stats from
   periodic snapshots of that buffer.

This is a **new, standalone file**, not a page added to the existing `skilleye-demo/app.py` --
that app is the validated, already-demoed pipeline (skeleton, stroke classification, quality
score); this is a separate, lower-stakes hardware-monitoring tool, and keeping them apart means
a bug here can't touch the thing that already works.

### What is and isn't tested

`LiveBuffer` (buffering/windowing logic) gets full unit tests. The Streamlit rendering and the
background-thread/socket wiring are not unit-tested -- consistent with this project's existing
convention (`skilleye-demo/app.py` has no test file either) and because there's no physical
racket connected in this environment to validate against. Instead, an **integration test**
spins up a local TCP server that replays the firmware's exact wire format (header comments +
CSV rows) and drives real `IMUStream` + `iter_rows` + `LiveBuffer` end to end -- proving the
whole read path works without needing the real device. Real end-to-end validation against the
physical racket happens on the user's machine.

## Components

### 1. `hardware/client/live_buffer.py` (new)

- `LiveBuffer(window_seconds=5.0)` -- `add_row(seq, t_us, values)` appends one
  `(ax,ay,az,gx,gy,gz)` sample, evicting samples older than `window_seconds` relative to the
  latest `t_us` seen so far.
- `snapshot()` -- returns `{"t_s": [...], "ax": [...], ..., "gz": [...]}`, `t_s` relative to
  the oldest sample currently in the window (so a chart's x-axis is stable/increasing), empty
  lists if no data yet.

### 2. `hardware/client/live_dashboard.py` (new)

- Sidebar: host/port (defaulting from `imu_client.load_config()`, same as the CLI), Connect
  button.
- On connect: a background `threading.Thread` opens `IMUStream`, iterates `iter_rows`, calls
  `buffer.add_row(...)` and updates a shared `StreamStats`, guarded by a lock; the Streamlit
  main thread polls a snapshot every ~150ms and redraws two charts (accel, gyro) plus
  Hz/dropped/peak-g metrics (reusing `_status_line`-style values from `StreamStats`).
- Connection errors (device off, wrong network) surface the same message
  `imu_client.py`'s CLI already prints, not a new one.
- Run with: `streamlit run live_dashboard.py` from `hardware/client/`.

## Out of scope

- Any change to `imu_client.py`'s CLI, `IMUStream`, `iter_rows`, or `StreamStats` -- reused
  as-is.
- Integrating into `skilleye-demo/app.py`.
- 3D orientation / racket-face-angle visualization (a separate, harder sensor-fusion problem,
  not attempted here).
