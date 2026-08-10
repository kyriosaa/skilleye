"""
Streamlit demo: pick a held-out sample swing, see its predicted stroke type,
quality score, per-phase/joint deviation table, and correction suggestions.

Deploy layout (self-contained — everything is relative to this file):
    skilleye-demo/
      app.py                 <- this file
      requirements.txt
      stroke_dataset.py
      stgcn_model.py
      quality/               (the package: score.py, phases.py, llm_explainer.py, ...)
      assets/
        best_model.pt        (from ml/results/stroke_classifier_v2/)
        templates.json       (from ml/results/quality_templates/)
      skeletons/             (validation-subject JSONs: 9,10,16,18,24,28,32,39,45,51,54)

Run locally:
    streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt

from stroke_dataset import load_records, STROKE_CLASSES, resample_time, add_velocity
from stgcn_model import STGCN, COCO17_EDGES
from quality.score import score_clip
from quality.llm_explainer import generate_explanation, LLMExplanationError

# --- paths are relative to this file, so it works locally and on Streamlit Cloud ---
BASE = Path(__file__).resolve().parent
SKELETONS_DIR = str(BASE / "skeletons")
TEMPLATES_PATH = str(BASE / "assets" / "templates.json")
STROKE_MODEL_PATH = str(BASE / "assets" / "best_model.pt")


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
    return data["templates"], set(data["val_subjects"])


@st.cache_data
def load_demo_records():
    _, val_subjects = load_templates()
    records, _ = load_records(SKELETONS_DIR)
    return [r for r in records if r["subject_id"] in val_subjects]


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

    templates, _ = load_templates()
    records = load_demo_records()

    # --- friendly empty-state instead of a crash if skeletons/ wasn't populated ---
    if not records:
        st.error(
            "No skeleton clips found in the `skeletons/` folder. Commit a subset of the "
            "validation-subject JSON files (subjects 9, 10, 16, 18, 24, 28, 32, 39, 45, "
            "51, 54) — see the deploy guide for the copy command."
        )
        st.stop()

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

    model = load_stroke_model()
    pred_stroke, pred_confidence = predict_stroke(model, kpts)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Skeleton viewer")
        frame_idx = st.slider("Frame", 0, kpts.shape[0] - 1, 0)
        st.pyplot(render_skeleton_frame(kpts, frame_idx))
        st.metric("Predicted stroke", pred_stroke, f"{pred_confidence*100:.0f}% confidence")
        st.caption(f"True label (for reference): {record['stroke']}, {record['skill_level']}")

    with col2:
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
