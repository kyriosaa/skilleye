"""
Builds per-stroke-class expert-motion templates for quality scoring: for each
stroke class, the mean and std of each joint's average angle within each
phase (backswing/contact/follow_through), computed across that class's
expert-labeled clips from the TRAINING side of this project's standard
subject-disjoint split (same split, same seed, as the single-split
classifiers elsewhere in this repo -- not a new split). The held-out
validation subjects from that same split are saved alongside the templates
so the demo UI can restrict itself to clips that were never used to build
the template being compared against.

Usage:
    python build_expert_templates.py --skeletons E:/SkillEye/skeletons \
        --out E:/SkillEye/ml/results/quality_templates/templates.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# stroke_dataset pulls in torch just by being imported (it defines a
# torch.utils.data.Dataset subclass) -- deferred into main() so the pure
# functions below (clip_joint_vector, compute_covariance_templates) stay
# importable and unit-testable on a machine without torch installed.
from quality.phases import split_phases, PHASES
from quality.angles import phase_mean_angles
from quality.correlation import (
    JOINT_ORDER, fit_covariance, shrink_correlation, shrinkage_intensity, shrinkage_target,
)


def clip_phase_means(kpts):
    """kpts: (T, 17, 2). Returns {phase: {joint: float or None}}."""
    phases = split_phases(kpts)
    return {phase: phase_mean_angles(phases[phase]) for phase in PHASES}


def clip_joint_vector(phase_kpts):
    """phase_kpts: (T, 17, 2) for a single phase. Returns a (len(JOINT_ORDER),)
    array in JOINT_ORDER order, or None if any tracked joint had no frames in
    this phase (mirrors phase_mean_angles's None-for-empty-phase contract)."""
    means = phase_mean_angles(phase_kpts)
    if any(means[joint] is None for joint in JOINT_ORDER):
        return None
    return np.array([means[joint] for joint in JOINT_ORDER])


def compute_covariance_templates(records, stroke_classes):
    """records: list of {"stroke": str, "kpts": (T, 17, 2)} for expert clips
    on the training side of the split (same shape as build_expert_templates'
    main() already assembles). stroke_classes: the fixed list of stroke names
    to always produce an (possibly empty) entry for.

    Returns covariance[stroke][phase] = {"joint_order", "mean", "cov", "n"} --
    the per-(stroke, phase) covariance matrix over JOINT_ORDER, shrunk toward
    shrinkage_target() with shrinkage_intensity(n, p) (Module A design:
    docs/superpowers/specs/2026-08-14-correlated-zscore-module-a-design.md).
    A (stroke, phase) with fewer than 2 usable clips is omitted -- not enough
    data to estimate a covariance at all -- rather than raising or silently
    inventing one."""
    vectors_by_stroke_phase = defaultdict(lambda: defaultdict(list))
    for r in records:
        phases = split_phases(r["kpts"])
        for phase in PHASES:
            vec = clip_joint_vector(phases[phase])
            if vec is not None:
                vectors_by_stroke_phase[r["stroke"]][phase].append(vec)

    target_corr = shrinkage_target()
    p = len(JOINT_ORDER)
    covariance = {}
    for stroke in stroke_classes:
        covariance[stroke] = {}
        for phase in PHASES:
            vectors = vectors_by_stroke_phase[stroke][phase]
            n = len(vectors)
            if n < 2:
                continue
            mean, cov = fit_covariance(np.array(vectors))
            shrunk = shrink_correlation(cov, target_corr, shrinkage_intensity(n, p))
            covariance[stroke][phase] = {
                "joint_order": JOINT_ORDER,
                "mean": mean.tolist(),
                "cov": shrunk.tolist(),
                "n": n,
            }
    return covariance


def main():
    from skeleton_records import load_records, subject_disjoint_split, STROKE_CLASSES

    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records, _ = load_records(args.skeletons)
    train_records, val_records, val_subjects = subject_disjoint_split(
        records, val_frac=args.val_frac, seed=args.seed)
    print(f"train: {len(train_records)} clips, val (held out, demo-eligible): "
          f"{len(val_records)} clips ({len(val_subjects)} subjects)")

    expert_train = [r for r in train_records if r["skill_level"] == "expert"]
    print(f"expert clips in training side: {len(expert_train)}")

    # values[stroke][phase][joint] = list of per-clip phase-mean angles
    values = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in expert_train:
        means = clip_phase_means(r["kpts"])
        for phase in PHASES:
            for joint, value in means[phase].items():
                if value is not None:
                    values[r["stroke"]][phase][joint].append(value)

    templates = {}
    for stroke in STROKE_CLASSES:
        templates[stroke] = {}
        for phase in PHASES:
            templates[stroke][phase] = {}
            joint_lists = values[stroke][phase]
            for joint, joint_values in joint_lists.items():
                arr = np.array(joint_values)
                templates[stroke][phase][joint] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "n": int(len(arr)),
                }
            n_clips = len(next(iter(joint_lists.values()), []))
            print(f"  {stroke:16s} {phase:16s}: {n_clips} expert clips contributed")

    covariance = compute_covariance_templates(expert_train, STROKE_CLASSES)
    for stroke in STROKE_CLASSES:
        for phase in PHASES:
            if phase in covariance[stroke]:
                print(f"  {stroke:16s} {phase:16s}: covariance from "
                      f"{covariance[stroke][phase]['n']} expert clips")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "templates": templates,
            "covariance": covariance,
            "val_subjects": sorted(val_subjects),
            "phases": PHASES,
        }, f, indent=2)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
