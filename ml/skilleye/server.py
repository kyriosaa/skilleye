import base64
import asyncio
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rtmlib import Body, Hand

from stroke_dataset import STROKE_CLASSES, resample_time, add_velocity
from stgcn_model import STGCN, COCO17_EDGES
from skeleton_pipeline import clean_clip
from quality.score import score_clip

HARDWARE_CLIENT_DIR = Path(__file__).resolve().parents[2] / "hardware" / "client"
if str(HARDWARE_CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(HARDWARE_CLIENT_DIR))
from imu_client import IMUStream, StreamStats, iter_rows, load_config

APP_DIR = Path(__file__).parent.resolve()
TEMPLATES_PATH = str(APP_DIR / "../results/quality_templates/templates.json")
STROKE_MODEL_PATH = str(APP_DIR / "../results/stroke_classifier_v2/best_model.pt")

app = FastAPI(title="SkillEye AI Analysis Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model caches
pose_body = None
hand_body = None
stroke_model = None
templates = None

STROKE_DISPLAY_MAP = {
    "forehand": "正手拍 (Forehand)",
    "backhand": "反手拍 (Backhand)",
    "serve": "發球 (Serve)",
    "smash": "高壓殺球 (Smash)",
    "forehand_volley": "正手截擊 (Forehand Volley)",
    "backhand_volley": "反手截擊 (Backhand Volley)",
}

IMU_CONFIG = load_config()
IMU_RECORDINGS_DIR = (APP_DIR / "../results/imu_recordings").resolve()
IMU_STROKE_ACCEL_THRESHOLD_G = 2.0
IMU_STROKE_GYRO_THRESHOLD_DPS = 150.0
imu_session = None
imu_session_lock = threading.Lock()


def pick_device():
    """Return 'cuda' if a working GPU backend exists, else 'cpu'."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class FramesPayload(BaseModel):
    frames: list[str]  # base64-encoded JPEG frames
    fps: float = 10.0
    stroke: str | None = None


def load_resources():
    global pose_body, hand_body, stroke_model, templates
    device = pick_device()

    if pose_body is None:
        print("Loading RTMPose estimator (device={})...".format(device))
        pose_body = Body(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)

    if hand_body is None:
        print("Loading RTMPose hand estimator (device={})...".format(device))
        hand_body = Hand(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)
    
    if stroke_model is None:
        print("Loading ST-GCN stroke model (device={})...".format(device))
        torch_device = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        stroke_model = STGCN(num_classes=len(STROKE_CLASSES))
        stroke_model.load_state_dict(torch.load(STROKE_MODEL_PATH, map_location=torch_device))
        stroke_model.to(torch_device)
        stroke_model.eval()

    if templates is None:
        print("Loading expert biomechanics templates...")
        with open(TEMPLATES_PATH) as f:
            data = json.load(f)
        templates = data["templates"]


@app.on_event("startup")
def startup_event():
    load_resources()


def pose_frames(frames, step=1):
    """Run RTMPose on a list of BGR frames. step>1 subsamples frames to speed up
    full-video analysis while still capturing the swing trajectory."""
    per_frame_kpts, per_frame_scores = [], []
    for idx, frame in enumerate(frames):
        if idx % step != 0:
            continue
        keypoints, scores = pose_body(frame)
        per_frame_kpts.append(np.asarray(keypoints))
        per_frame_scores.append(np.asarray(scores))
    return per_frame_kpts, per_frame_scores


def check_motion_intensity(kpts, threshold=10.0):
    """Checks if the clip has significant sudden acceleration in arm joints (characteristic of a stroke)."""
    # Key joints for tennis stroke: Shoulder (5,6), Elbow (7,8), Wrist (9,10)
    critical_indices = [5, 6, 7, 8, 9, 10]
    kpts_critical = kpts[:, critical_indices, :]
    
    # Calculate acceleration (second derivative of position)
    vel = np.diff(kpts_critical, axis=0) # (T-1, V, 2)
    acc = np.diff(vel, axis=0)           # (T-2, V, 2)
    
    # Peak acceleration magnitude across these joints
    acc_mag = np.linalg.norm(acc, axis=2) # (T-2, V)
    peak_acc = np.max(acc_mag)
    return peak_acc > threshold


def analyze_kpts(per_frame_kpts, per_frame_scores, override_stroke=None):
    # ... (前略)
        
    # NEW: Check if motion intensity is significant (explosive acceleration)
    try:
        if not check_motion_intensity(np.asarray(per_frame_kpts)):
            raise HTTPException(
                status_code=400,
                detail="動作幅度或爆發力不足，請確認是否為標準網球揮拍動作。"
            )
    except HTTPException:
        raise
    except Exception:
        # Not enough frames or invalid keypoints -> skip motion-intensity gate
        pass

    # Clean the clip (single-subject tracking, resample to fixed length)
    kpts_arr, scores_arr, ok = clean_clip(per_frame_kpts, per_frame_scores)
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Primary subject visible in too few frames. Please use a clearer video of a single player."
        )

    # Predict stroke (or use the user-selected stroke in no-sensor mode)
    if override_stroke:
        pred_stroke = override_stroke
        pred_confidence = 1.0
    else:
        resampled = resample_time(kpts_arr, 64)
        with_velocity = add_velocity(resampled)
        tensor = torch.from_numpy(with_velocity.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        device = next(stroke_model.parameters()).device
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = stroke_model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(probs.argmax())
        pred_stroke = STROKE_CLASSES[pred_idx]
        pred_confidence = float(probs[pred_idx])

    # Score against expert biomechanics templates
    result = score_clip(kpts_arr, pred_stroke, templates)

    # NEW: Check stroke prediction confidence threshold (Strict: 0.85)
    STROKE_CONFIDENCE_THRESHOLD = 0.85 
    if not override_stroke and pred_confidence < STROKE_CONFIDENCE_THRESHOLD:
        raise HTTPException(
            status_code=400,
            detail=f"動作不明確 (信心值 {pred_confidence:.2f})，請確認是否為揮拍動作。"
        )


    return {
        "overall_score": float(result["overall_score"]),
        "predicted_stroke": pred_stroke,
        "predicted_stroke_display": STROKE_DISPLAY_MAP.get(pred_stroke, pred_stroke),
        "predicted_confidence": pred_confidence,
        "table": result["table"],
        "suggestions": result["suggestions"],
    }


@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    load_resources()

    suffix = Path(file.filename).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        frames, _ = read_video_frames(tmp_path)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not decode frames.")
        # Subsample to 1 of every 3 frames for speed (GPU: ~0.12s/frame)
        per_frame_kpts, per_frame_scores = pose_frames(frames, step=3)
        return analyze_kpts(per_frame_kpts, per_frame_scores)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)


def _imu_connection():
    return IMUStream(
        IMU_CONFIG.get("host", "192.168.4.1"),
        int(IMU_CONFIG.get("port", 3333)),
        timeout=10.0,
    )


@app.get("/imu/status")
async def imu_status():
    """Check whether the ESP32 IMU accepts a TCP connection."""
    try:
        with _imu_connection():
            return {"connected": True}
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        return {"connected": False, "error": str(exc)}


def _record_imu(seconds, output_path):
    stats = StreamStats()
    started = time.monotonic()
    with _imu_connection() as stream, open(output_path, "w", encoding="utf-8", newline="") as fh:
        for line in stream.lines():
            fh.write(line + "\n")
            for _ in iter_rows([line], stats):
                pass
            if time.monotonic() - started >= seconds:
                break
    return stats


def _imu_session_worker(session):
    try:
        with _imu_connection() as stream:
            session["connected"] = True
            session["ready"].set()
            with open(session["output_path"], "w", encoding="utf-8", newline="") as fh:
                for line in stream.lines():
                    if session["stop"].is_set():
                        break
                    fh.write(line + "\n")
                    for _, _, values in iter_rows([line], session["stats"]):
                        accel = (values[0] ** 2 + values[1] ** 2 + values[2] ** 2) ** 0.5
                        gyro = (values[3] ** 2 + values[4] ** 2 + values[5] ** 2) ** 0.5
                        session["peak_accel_g"] = max(session["peak_accel_g"], accel)
                        session["peak_gyro_dps"] = max(session["peak_gyro_dps"], gyro)
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        session["error"] = str(exc)
        session["ready"].set()


@app.post("/imu/session/start")
async def start_imu_session():
    """Start collecting IMU data for the browser's paired recording."""
    global imu_session
    with imu_session_lock:
        if imu_session is not None:
            raise HTTPException(status_code=409, detail="An IMU recording is already active")
        IMU_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = IMU_RECORDINGS_DIR / f"browser_{int(time.time() * 1_000_000)}.csv"
        session = {
            "stop": threading.Event(), "ready": threading.Event(),
            "stats": StreamStats(), "output_path": output_path,
            "connected": False, "error": None,
            "peak_accel_g": 0.0, "peak_gyro_dps": 0.0,
        }
        imu_session = session
        session["thread"] = threading.Thread(
            target=_imu_session_worker, args=(session,), daemon=True)
        session["thread"].start()

    await asyncio.to_thread(session["ready"].wait, 10.0)
    if session["error"]:
        with imu_session_lock:
            imu_session = None
        raise HTTPException(status_code=503, detail=f"IMU connection failed: {session['error']}")
    return {"connected": session["connected"]}


@app.post("/imu/session/stop")
async def stop_imu_session():
    """Stop the paired IMU recording and report whether a stroke was detected."""
    global imu_session
    with imu_session_lock:
        session = imu_session
        imu_session = None
    if session is None:
        raise HTTPException(status_code=404, detail="No active IMU recording")

    session["stop"].set()
    await asyncio.to_thread(session["thread"].join, 10.0)
    stroke_detected = (
        session["peak_accel_g"] >= IMU_STROKE_ACCEL_THRESHOLD_G
        or session["peak_gyro_dps"] >= IMU_STROKE_GYRO_THRESHOLD_DPS
    )
    return {
        "stroke_detected": stroke_detected,
        "peak_accel_g": session["peak_accel_g"],
        "peak_gyro_dps": session["peak_gyro_dps"],
        "rows": session["stats"].rows,
        "missing": session["stats"].missing,
        "recording": str(session["output_path"].relative_to(APP_DIR.parent.parent)),
    }


@app.post("/imu/record")
async def record_imu(
    seconds: float = Query(10.0, gt=0, le=300),
    filename: str = Query("imu_recording.csv"),
):
    """Record raw ESP32 IMU CSV data for a bounded duration."""
    safe_filename = Path(filename).name
    if not safe_filename.lower().endswith(".csv"):
        safe_filename += ".csv"
    IMU_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = IMU_RECORDINGS_DIR / safe_filename
    try:
        stats = await asyncio.to_thread(_record_imu, seconds, output_path)
    except (ConnectionRefusedError, OSError, TimeoutError) as exc:
        raise HTTPException(status_code=503, detail=f"IMU connection failed: {exc}") from exc

    return {
        "recording": str(output_path.relative_to(APP_DIR.parent.parent)),
        "rows": stats.rows,
        "missing": stats.missing,
        "duration_s": stats.duration_s,
        "measured_hz": stats.measured_hz,
        "metadata": stats.metadata,
    }


@app.post("/analyze_frame")
async def analyze_frame(image: UploadFile = File(...)):
    load_resources()
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(400, "Decode error")

    keypoints, scores = pose_body(frame)
    kpts = np.asarray(keypoints); sc = np.asarray(scores)
    persons = []
    if kpts.ndim == 3 and kpts.shape[0] > 0:
        for p in range(kpts.shape[0]):
            persons.append([
                {"x": float(kpts[p, j, 0]), "y": float(kpts[p, j, 1]), "score": float(sc[p, j])}
                for j in range(kpts.shape[1])
            ])
    return {"persons": persons}


# MediaPipe/RTMPose hand landmark indices
_HAND_FINGER_TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
_HAND_FINGER_PIPS = {"thumb": 2, "index": 6, "middle": 10, "ring": 14, "pinky": 18} # Use MCP (2) for thumb


def _finger_extended(kpts, tip_idx, pip_idx, min_score=0.3):
    """A finger counts as extended when its tip is clearly farther from the
    wrist than its PIP joint -- i.e. the finger is straight rather than bent."""
    if kpts[tip_idx, 2] < min_score or kpts[pip_idx, 2] < min_score:
        return False
    wrist = kpts[0, :2]
    d_tip = float(np.linalg.norm(kpts[tip_idx, :2] - wrist))
    d_pip = float(np.linalg.norm(kpts[pip_idx, :2] - wrist))
    # Reduced multiplier to 1.15 for more natural and robust open-palm detection
    return d_tip > d_pip * 1.15


def is_five_gesture(kpts):
    """True when all five fingers of a hand are extended (open palm)."""
    return all(
        _finger_extended(kpts, _HAND_FINGER_TIPS[f], _HAND_FINGER_PIPS[f])
        for f in _HAND_FINGER_TIPS
    )


@app.post("/detect_gesture")
async def detect_gesture(image: UploadFile = File(...)):
    load_resources()
    data = await image.read()
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None: raise HTTPException(400, "Decode error")

    keypoints, scores = hand_body(frame)
    kpts = np.asarray(keypoints); sc = np.asarray(scores)
    hands = []
    if kpts.ndim == 3 and kpts.shape[0] > 0:
        for p in range(kpts.shape[0]):
            mean_score = float(sc[p].mean())
            if mean_score < 0.6:  # Reject low-confidence hands / false positives
                continue
            hand_kpts = np.column_stack([kpts[p], sc[p]])
            hands.append({
                "five": bool(is_five_gesture(hand_kpts)),
                "score": mean_score,
            })
    return {"hands": hands, "five": any(h["five"] for h in hands)}


@app.post("/analyze_frames")
async def analyze_frames(payload: FramesPayload):
    load_resources()
    frames = []
    for b64 in payload.frames:
        raw = base64.b64decode(b64.split(",")[-1])
        arr = np.frombuffer(raw, np.uint8)
        f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if f is not None: frames.append(f)
    if not frames: raise HTTPException(400, "No frames")
    per_frame_kpts, per_frame_scores = pose_frames(frames)
    return analyze_kpts(per_frame_kpts, per_frame_scores, override_stroke=payload.stroke)


def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames, fps


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)