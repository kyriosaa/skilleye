"""
Runs the full quality-scoring evaluation (Module A independent + correlated,
Module B where applicable) on the real webcam+IMU recordings, and renders
skeleton + sensor figures for the report. Ad-hoc analysis script for this
one batch of real recordings -- not part of the tested library (same
convention as smoke_check_quality_scoring.py).

Usage:
    python evaluate_real_recordings.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hardware" / "client"))

from stgcn_model import COCO17_EDGES
from quality.score import score_clip, score_clip_correlated
from quality.skill_rules import (
    check_volley_swing_effort, evaluate_backhand_volley_skill_rules, g_to_mps2,
)
from imu_client import iter_rows, StreamStats

REPO_ROOT = Path(__file__).resolve().parents[2]
SKELETONS_DIR = REPO_ROOT / "hardware" / "client" / "newresult_skeletons"
RECORDINGS_DIR = REPO_ROOT / "hardware" / "client" / "newresult"
TEMPLATES_PATH = REPO_ROOT / "ml" / "results" / "quality_templates" / "templates.json"
OUT_DIR = REPO_ROOT / "hardware" / "client" / "newresult_eval"

STROKE_NAME_MAP = {
    "backhand": "backhand", "forehand": "forehand",
    "backhandvolley": "backhand_volley", "forehandvolley": "forehand_volley",
    "serve": "serve", "smash": "smash",
}


def load_skeleton(stroke_file_stem):
    with open(SKELETONS_DIR / f"{stroke_file_stem}.json") as f:
        rec = json.load(f)
    kpts = np.array(rec["keypoints_normalized"], dtype=np.float32)
    return kpts, rec


def render_skeleton_png(kpts, frame_idx, out_path, title):
    frame = kpts[frame_idx]
    fig, ax = plt.subplots(figsize=(4, 4))
    for a, b in COCO17_EDGES:
        ax.plot([frame[a, 0], frame[b, 0]], [-frame[a, 1], -frame[b, 1]],
                color="#2a78d6", linewidth=2)
    ax.scatter(frame[:, 0], -frame[:, 1], color="#184f95", s=15, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def imu_stats_and_plot(imu_path, out_path, title):
    with open(imu_path, encoding="utf-8") as f:
        lines = f.readlines()
    stats = StreamStats()
    rows = list(iter_rows(lines, stats))
    t0 = rows[0][1]
    t_s = [(t_us - t0) / 1e6 for _, t_us, _ in rows]
    accs = [math.sqrt(ax * ax + ay * ay + az * az) for _, _, (ax, ay, az, gx, gy, gz) in rows]
    gyro_mag = [math.sqrt(gx * gx + gy * gy + gz * gz) for _, _, (ax, ay, az, gx, gy, gz) in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
    ax1.plot(t_s, accs, color="#e76f51", linewidth=0.8)
    ax1.set_ylabel("|accel| (g)")
    ax1.set_title(title, fontsize=10)
    ax2.plot(t_s, gyro_mag, color="#2a9d8f", linewidth=0.8)
    ax2.set_ylabel("|gyro| (deg/s)")
    ax2.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

    peak_g = max(accs)
    peak_idx = accs.index(peak_g)
    return {"rows": stats.rows, "duration_s": t_s[-1], "peak_g": peak_g,
            "peak_t_s": t_s[peak_idx], "peak_mps2": g_to_mps2(peak_g)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(TEMPLATES_PATH) as f:
        template_data = json.load(f)
    templates = template_data["templates"]
    covariance = template_data.get("covariance", {})

    results = {}
    for file_stem, stroke in STROKE_NAME_MAP.items():
        kpts, rec = load_skeleton(file_stem)

        # contact frame: peak dominant-wrist speed, same heuristic as quality/phases.py
        from quality.phases import detect_contact_frame
        contact = detect_contact_frame(kpts)

        render_skeleton_png(
            kpts, contact, OUT_DIR / f"{file_stem}_skeleton.png",
            f"{stroke} -- contact frame ({contact}/{kpts.shape[0]})")

        independent = score_clip(kpts, stroke, templates)
        correlated = None
        if stroke in covariance and covariance[stroke]:
            correlated = score_clip_correlated(kpts, stroke, covariance)

        skill_flags = None
        if stroke == "backhand_volley":
            skill_flags = evaluate_backhand_volley_skill_rules(kpts)["flags"]

        imu_path = RECORDINGS_DIR / f"{file_stem}_imu.csv"
        imu_summary = imu_stats_and_plot(
            imu_path, OUT_DIR / f"{file_stem}_imu.png", f"{stroke} -- IMU (real recording)")

        volley_effort = None
        if stroke in ("backhand_volley", "forehand_volley"):
            volley_effort = check_volley_swing_effort(imu_summary["peak_mps2"])

        results[stroke] = {
            "file_stem": file_stem,
            "num_frames": rec["num_frames"], "valid_frac": rec["valid_frac"],
            "contact_frame": contact,
            "independent_score": independent["overall_score"],
            "correlated_score": correlated["overall_score"] if correlated else None,
            "skill_flags": skill_flags,
            "imu": imu_summary,
            "volley_effort": volley_effort,
        }
        print(f"{stroke:16s} indep={independent['overall_score']:.0f} "
              f"corr={correlated['overall_score'] if correlated else float('nan'):.0f} "
              f"peak={imu_summary['peak_g']:.2f}g flags={skill_flags}")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT_DIR / 'results.json'} and figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
