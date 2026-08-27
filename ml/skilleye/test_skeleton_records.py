"""Tests for skeleton_records.py -- extracted from stroke_dataset.py so this
pure data-loading/splitting logic is importable without torch (stroke_dataset.py
also defines a torch.utils.data.Dataset subclass, which pulls in torch just by
being imported)."""
import json

import numpy as np
import pytest

from skeleton_records import (
    CATEGORY_TO_STROKE,
    STROKE_CLASSES,
    STROKE_TO_IDX,
    load_records,
    subject_disjoint_split,
    subject_kfold_split,
)


def make_record(subject_id, skill_level, stroke="backhand"):
    return {"subject_id": subject_id, "skill_level": skill_level, "stroke": stroke}


class TestSubjectDisjointSplit:

    def test_no_subject_appears_in_both_train_and_val(self):
        records = ([make_record(i, "beginner") for i in range(1, 21)]
                   + [make_record(i, "expert") for i in range(21, 41)])
        train, val, val_subjects = subject_disjoint_split(records, val_frac=0.2, seed=42)

        train_subjects = {r["subject_id"] for r in train}
        val_subjects_from_records = {r["subject_id"] for r in val}
        assert train_subjects.isdisjoint(val_subjects_from_records)
        assert val_subjects_from_records == val_subjects

    def test_val_set_keeps_a_mix_of_both_skill_levels(self):
        records = ([make_record(i, "beginner") for i in range(1, 21)]
                   + [make_record(i, "expert") for i in range(21, 41)])
        _, val, _ = subject_disjoint_split(records, val_frac=0.2, seed=42)

        val_levels = {r["skill_level"] for r in val}
        assert val_levels == {"beginner", "expert"}

    def test_is_deterministic_given_the_same_seed(self):
        records = [make_record(i, "beginner") for i in range(1, 11)]
        _, _, val_a = subject_disjoint_split(records, seed=7)
        _, _, val_b = subject_disjoint_split(records, seed=7)
        assert val_a == val_b


class TestSubjectKfoldSplit:

    def test_every_subject_is_held_out_exactly_once_across_folds(self):
        records = ([make_record(i, "beginner") for i in range(1, 11)]
                   + [make_record(i, "expert") for i in range(11, 21)])
        folds = subject_kfold_split(records, k=5, seed=42)

        assert len(folds) == 5
        all_val_subjects = [s for _, _, val_subjects in folds for s in val_subjects]
        assert sorted(all_val_subjects) == list(range(1, 21))  # each exactly once

    def test_each_folds_train_and_val_subjects_are_disjoint(self):
        records = [make_record(i, "beginner") for i in range(1, 11)]
        folds = subject_kfold_split(records, k=5, seed=42)
        for train, val, val_subjects in folds:
            train_subjects = {r["subject_id"] for r in train}
            assert train_subjects.isdisjoint(val_subjects)


class TestLoadRecords:

    def test_reads_json_files_and_merges_categories(self, tmp_path):
        (tmp_path / "backhand").mkdir()
        (tmp_path / "backhand_volley").mkdir()

        def write(path, category, subject_id, skill_level):
            data = {
                "category": category, "subject_id": subject_id, "skill_level": skill_level,
                "keypoints_normalized": np.zeros((3, 17, 2)).tolist(),
            }
            with open(path, "w") as f:
                json.dump(data, f)

        write(tmp_path / "backhand" / "p1_backhand_s1.json", "backhand", 1, "beginner")
        write(tmp_path / "backhand_volley" / "p2_bv_s1.json", "backhand_volley", 2, "expert")

        records, skipped = load_records(tmp_path)

        assert skipped == 0
        assert len(records) == 2
        strokes = {r["stroke"] for r in records}
        assert strokes == {"backhand", "backhand_volley"}
        assert records[0]["kpts"].shape == (3, 17, 2)

    def test_skips_unrecognized_categories_without_raising(self, tmp_path):
        (tmp_path / "unknown_category").mkdir()
        data = {"category": "unknown_category", "subject_id": 1, "skill_level": "beginner",
                "keypoints_normalized": np.zeros((2, 17, 2)).tolist()}
        with open(tmp_path / "unknown_category" / "x.json", "w") as f:
            json.dump(data, f)

        records, skipped = load_records(tmp_path)

        assert records == []
        assert skipped == 1


def test_category_to_stroke_and_stroke_classes_are_consistent():
    # Every CATEGORY_TO_STROKE target must be a real stroke class, and every
    # stroke class must be reachable from at least one category -- otherwise
    # STROKE_TO_IDX or load_records would silently mis-handle a category.
    assert set(CATEGORY_TO_STROKE.values()) == set(STROKE_CLASSES)
    assert set(STROKE_TO_IDX.keys()) == set(STROKE_CLASSES)
