"""
Runs RTMPose over the real webcam recordings from hardware/client/newresult/
(one-off, ad-hoc video filenames -- not THETIS's batch naming convention, so
this reuses skeleton_pipeline.clean_clip directly rather than going through
batch_extract.py's THETIS-specific file-pattern parsing).

Usage:
    python extract_real_recordings.py --src ../../hardware/client/newresult \
        --dst ../../hardware/client/newresult_skeletons
"""
import argparse
import json
from pathlib import Path

import torch  # noqa: F401  (import before onnxruntime so its CUDA/cuDNN DLLs are on the search path)
import cv2
import numpy as np
from rtmlib import Body
from rtmlib.tools import base as rtmlib_base

rtmlib_base.RTMLIB_SETTINGS["onnxruntime"]["directml"] = "DmlExecutionProvider"

from skeleton_pipeline import clean_clip


def run_pose_on_video(body, video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    per_frame_kpts, per_frame_scores = [], []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        keypoints, scores = body(frame)
        per_frame_kpts.append(np.asarray(keypoints))
        per_frame_scores.append(np.asarray(scores))
    cap.release()
    return per_frame_kpts, per_frame_scores, fps, w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "directml", "cpu"])
    args = ap.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    body = Body(mode="lightweight", to_openpose=False, backend="onnxruntime", device=args.device)

    videos = sorted(src_root.glob("*_video.mp4"))
    print(f"found {len(videos)} videos in {src_root}")

    for video_path in videos:
        stroke = video_path.stem.replace("_video", "")
        out_path = dst_root / f"{stroke}.json"
        if out_path.exists():
            print(f"  [skip] {stroke} (already extracted)")
            continue

        per_frame_kpts, per_frame_scores, fps, w, h = run_pose_on_video(body, video_path)
        kpts, scores, ok = clean_clip(per_frame_kpts, per_frame_scores)
        valid_frac = float((scores >= 0.3).mean())
        print(f"  {stroke}: {len(per_frame_kpts)} frames, valid_frac={valid_frac:.2f}, ok={ok}")

        record = {
            "source_video": str(video_path),
            "stroke": stroke,
            "fps": fps,
            "width": w,
            "height": h,
            "num_frames": int(kpts.shape[0]),
            "valid_frac": valid_frac,
            "ok": ok,
            "keypoints_normalized": kpts.tolist(),
            "keypoint_scores": scores.tolist(),
        }
        with open(out_path, "w") as f:
            json.dump(record, f)

    print("done")


if __name__ == "__main__":
    main()
