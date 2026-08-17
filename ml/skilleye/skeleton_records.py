"""
Skeleton-JSON loading and subject-level splitting -- extracted from
stroke_dataset.py so this pure I/O/numpy logic is importable without torch.
stroke_dataset.py also defines a torch.utils.data.Dataset subclass, which
pulls in torch just by being imported, even for callers (like
build_expert_templates.py) that only need the loading/splitting below.
stroke_dataset.py re-exports everything here for backward compatibility --
existing `from stroke_dataset import load_records, ...` callers are unaffected.
"""
import json
from pathlib import Path

import numpy as np

CATEGORY_TO_STROKE = {
    "backhand": "backhand",
    "backhand2hands": "backhand",
    "backhand_slice": "backhand",
    "backhand_volley": "backhand_volley",
    "forehand_flat": "forehand",
    "forehand_openstands": "forehand",
    "forehand_slice": "forehand",
    "forehand_volley": "forehand_volley",
    "flat_service": "serve",
    "kick_service": "serve",
    "slice_service": "serve",
    "smash": "smash",
}
STROKE_CLASSES = ["backhand", "forehand", "backhand_volley", "forehand_volley", "serve", "smash"]
STROKE_TO_IDX = {s: i for i, s in enumerate(STROKE_CLASSES)}


def load_records(skeleton_root):
    """Walk the skeleton JSON tree once, merge categories, return raw records
    (kept in memory as plain arrays; the full extracted set is small enough)."""
    root = Path(skeleton_root)
    records = []
    skipped_category = 0
    for json_path in sorted(root.rglob("*.json")):
        with open(json_path) as f:
            d = json.load(f)
        category = d["category"]
        stroke = CATEGORY_TO_STROKE.get(category)
        if stroke is None:
            skipped_category += 1
            continue
        kpts = np.asarray(d["keypoints_normalized"], dtype=np.float32)  # (T, V, 2)
        records.append({
            "kpts": kpts,
            "label": STROKE_TO_IDX[stroke],
            "stroke": stroke,
            "subject_id": d["subject_id"],
            "skill_level": d["skill_level"],
            "source": str(json_path),
        })
    return records, skipped_category


def subject_disjoint_split(records, val_frac=0.2, seed=42):
    """Split by subject id so no subject appears in both train and val.
    Stratifies the subject-level split by skill_level so both splits keep a
    beginner/expert mix (THETIS's beginner/expert split is subject-level:
    p1-p31 beginner, p32-p55 expert)."""
    rng = np.random.RandomState(seed)

    subjects_by_level = {"beginner": set(), "expert": set()}
    for r in records:
        subjects_by_level[r["skill_level"]].add(r["subject_id"])

    val_subjects = set()
    for level, subjects in subjects_by_level.items():
        subjects = sorted(subjects)
        rng.shuffle(subjects)
        n_val = max(1, int(round(len(subjects) * val_frac)))
        val_subjects.update(subjects[:n_val])

    train = [r for r in records if r["subject_id"] not in val_subjects]
    val = [r for r in records if r["subject_id"] in val_subjects]
    return train, val, val_subjects


def subject_kfold_split(records, k=5, seed=42):
    """k-fold split by subject id, stratified by skill_level so every fold keeps a
    beginner/expert mix. Returns a list of k (train_records, val_records, val_subjects)
    tuples -- one held-out validation number from a single split can be a lucky (or
    unlucky) draw of which subjects it happened to hold out; reporting mean/std across
    folds is the honest, defensible number."""
    rng = np.random.RandomState(seed)

    subjects_by_level = {"beginner": set(), "expert": set()}
    for r in records:
        subjects_by_level[r["skill_level"]].add(r["subject_id"])

    fold_subjects = [set() for _ in range(k)]
    for level, subjects in subjects_by_level.items():
        subjects = sorted(subjects)
        rng.shuffle(subjects)
        for i, s in enumerate(subjects):
            fold_subjects[i % k].add(s)

    folds = []
    for i in range(k):
        val_subjects = fold_subjects[i]
        train = [r for r in records if r["subject_id"] not in val_subjects]
        val = [r for r in records if r["subject_id"] in val_subjects]
        folds.append((train, val, val_subjects))
    return folds
