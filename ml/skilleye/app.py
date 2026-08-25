"""
Streamlit demo: pick a held-out sample swing, see its predicted stroke type,
quality score, per-phase/joint deviation table, and correction suggestions.

Run (from the skilleye/ directory):
    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
import os
import cv2
import tempfile
from rtmlib import Body

from stroke_dataset import load_records, STROKE_CLASSES, resample_time, add_velocity
from stgcn_model import STGCN, COCO17_EDGES
from quality.score import score_clip, score_clip_correlated
from quality.skill_rules import evaluate_backhand_volley_skill_rules
from quality.llm_explainer import generate_explanation, LLMExplanationError
from skeleton_pipeline import clean_clip

# Relative to this file, not a hardcoded drive letter, so the demo runs on
# whatever machine/drive the repo happens to be cloned to.
APP_DIR = Path(__file__).parent.resolve()
SKELETONS_DIR = str(APP_DIR.parent.parent / "skeletons")
TEMPLATES_PATH = str(APP_DIR.parent / "results/quality_templates/templates.json")
STROKE_MODEL_PATH = str(APP_DIR.parent / "results/stroke_classifier_v2/best_model.pt")


@st.cache_resource
def load_stroke_model():
    model = STGCN(num_classes=len(STROKE_CLASSES))
    model.load_state_dict(torch.load(STROKE_MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


@st.cache_resource
def load_templates():
    with open(TEMPLATES_PATH) as f:
        data = json.load(f)
    # covariance (Module A) is additive to the original templates.json shape --
    # absent on an older templates.json that predates build_expert_templates.py's
    # compute_covariance_templates(), so the correlated-scoring toggle degrades
    # gracefully (see main()) rather than KeyErroring.
    return data["templates"], data.get("covariance", {}), set(data["val_subjects"])


@st.cache_resource
def load_pose_estimator():
    # Load RTMPose body model
    return Body(mode="lightweight", to_openpose=False, backend="onnxruntime", device="cpu")


def run_pose_on_uploaded_video(body, video_file):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_file.read() if hasattr(video_file, 'read') else open(video_file, 'rb').read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
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
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    return per_frame_kpts, per_frame_scores, fps, w, h


@st.cache_data
def load_demo_records():
    try:
        _, _, val_subjects = load_templates()
        if not os.path.exists(SKELETONS_DIR):
            return []
        records = load_records(SKELETONS_DIR)
        return [r for r in records if r["subject_id"] in val_subjects]
    except Exception:
        return []


def predict_stroke(model, kpts):
    resampled = resample_time(kpts, 64)
    with_velocity = add_velocity(resampled)
    tensor = torch.from_numpy(with_velocity.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    return STROKE_CLASSES[pred_idx], float(probs[pred_idx])


@st.cache_data
def cached_llm_explanation(stroke, table):
    return generate_explanation(stroke, table)


def render_skeleton_frame(kpts, frame_idx):
    frame = kpts[frame_idx]
    fig, ax = plt.subplots(figsize=(4, 4))
    for a, b in COCO17_EDGES:
        ax.plot([frame[a, 0], frame[b, 0]], [-frame[a, 1], -frame[b, 1]], color="#2a78d6", linewidth=2)
    ax.scatter(frame[:, 0], -frame[:, 1], color="#184f95", s=15, zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def main():
    st.set_page_config(page_title="SkillEye Quality Scoring Demo", layout="wide")
    st.title("SkillEye: Swing Quality Scoring Demo")
    st.caption("Sample clips are held-out validation subjects only -- never used to "
               "build the expert templates being compared against.")

    templates, covariance, _ = load_templates()
    records = load_demo_records()

    video_input_source = st.sidebar.radio("Video Input Source", ["Upload Custom Video", "Select Sample Skeletons"])

    scoring_mode = st.sidebar.radio(
        "Scoring mode", ["Independent (v1)", "Correlated (Module A)"],
        help="Independent: each joint z-scored against its own template value alone. "
             "Correlated: each joint z-scored against its expected value given what the "
             "other joints are doing in the same phase (README §2.8).")

    kpts = None
    true_label_ref = "N/A"

    if video_input_source == "Upload Custom Video":
        st.subheader("Analyze Custom Swing Video")

        # Look for local videos as presets
        video_dir = APP_DIR.parent.parent.parent / "Video"
        local_videos = []
        if video_dir.exists():
            local_videos = sorted(list(video_dir.glob("*.mp4")))

        selected_preset = None
        if local_videos:
            preset_names = ["-- None --"] + [v.name for v in local_videos]
            preset_choice = st.selectbox("Or choose a built-in demo video", preset_names)
            if preset_choice != "-- None --":
                selected_preset = video_dir / preset_choice

        uploaded_file = st.file_uploader("Upload a swing video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

        video_to_process = uploaded_file or selected_preset

        if video_to_process:
            st.video(video_to_process)
            if st.button("Run AI Analysis 🤖"):
                with st.spinner("Extracting body skeleton and analyzing dynamics..."):
                    try:
                        pose_body = load_pose_estimator()
                        per_frame_kpts, per_frame_scores, fps, w, h = run_pose_on_uploaded_video(pose_body, video_to_process)
                        processed_kpts, scores, ok = clean_clip(per_frame_kpts, per_frame_scores)
                        if not ok:
                            st.error("⚠️ Primary subject visible in too few frames. Please use a clearer video of a single player.")
                        else:
                            st.session_state["custom_kpts"] = processed_kpts
                            st.session_state["custom_video_analyzed"] = True
                            st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"Error during pose extraction: {e}")

        if "custom_kpts" in st.session_state:
            kpts = st.session_state["custom_kpts"]
            true_label_ref = "Uploaded Video"

    else:
        if not records:
            st.warning("⚠️ No pre-extracted skeletons found. Please upload a custom video using the sidebar source option instead.")
        else:
            strokes_available = sorted({r["stroke"] for r in records})
            stroke_choice = st.sidebar.selectbox("Stroke category", strokes_available)

            clips_in_stroke = [r for r in records if r["stroke"] == stroke_choice]
            clip_labels = [
                f"subject {r['subject_id']} ({r['skill_level']}) - {Path(r['source']).stem}"
                for r in clips_in_stroke
            ]
            clip_idx = st.sidebar.selectbox(
                "Sample clip", range(len(clip_labels)), format_func=lambda i: clip_labels[i])
            record = clips_in_stroke[clip_idx]
            kpts = record["kpts"]
            true_label_ref = f"{record['stroke']}, {record['skill_level']}"

    if kpts is not None:
        model = load_stroke_model()
        pred_stroke, pred_confidence = predict_stroke(model, kpts)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Skeleton viewer")
            frame_idx = st.slider("Frame", 0, kpts.shape[0] - 1, 0)
            st.pyplot(render_skeleton_frame(kpts, frame_idx))
            st.metric("Predicted stroke", pred_stroke, f"{pred_confidence*100:.0f}% confidence")
            st.caption(f"True/Source label (for reference): {true_label_ref}")

        with col2:
            if scoring_mode == "Correlated (Module A)":
                if pred_stroke in covariance and covariance[pred_stroke]:
                    result = score_clip_correlated(kpts, pred_stroke, covariance)
                else:
                    st.warning(
                        f"No covariance data for '{pred_stroke}' in this templates.json "
                        "(rebuild with build_expert_templates.py) -- falling back to "
                        "independent scoring for this clip.")
                    result = score_clip(kpts, pred_stroke, templates)
            else:
                result = score_clip(kpts, pred_stroke, templates)

            st.subheader("Quality score")
            st.metric("Overall", f"{result['overall_score']:.0f} / 100")

            st.subheader("Per-phase / per-joint deviation")
            st.table([
                {
                    "phase": row["phase"],
                    "joint": row["joint"],
                    "value (rad)": round(row["value"], 3),
                    "z-score": round(row["z"], 2) if row["z"] is not None else row["note"],
                    "flagged": "!" if row["flagged"] else "",
                }
                for row in result["table"]
            ])

            st.subheader("Correction suggestions")
            if result["suggestions"]:
                for s in result["suggestions"]:
                    st.write(f"- {s}")
            else:
                st.write("No significant deviations flagged against the expert template.")

            if pred_stroke == "backhand_volley":
                st.subheader("Skill-level checks (Module B)")
                st.caption("Rules from Katsumi et al. (2026), comparing skilled vs. "
                           "less-skilled backhand volleys directly -- separate from the "
                           "expert-only z-score above (README §2.9).")
                skill_flags = evaluate_backhand_volley_skill_rules(kpts)["flags"]
                if skill_flags:
                    for flag in skill_flags:
                        st.write(f"- {flag['note']}")
                else:
                    st.write("No skill-level pattern flagged for this clip.")

            if st.button("Generate AI explanation"):
                try:
                    explanation = cached_llm_explanation(pred_stroke, result["table"])
                    st.info(explanation)
                except LLMExplanationError as e:
                    st.warning(
                        "AI explanation unavailable -- showing the rule-based suggestions "
                        f"above instead. ({e})")


if __name__ == "__main__":
    main()
