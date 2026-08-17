"""
Dataset for THETIS skeleton JSONs -> stroke-type classification.

v2: 6 classes instead of 5 -- forehand_volley and backhand_volley are kept
separate rather than merged into one "volley" class, since they're visually
distinct motions and merging them was the classifier's weakest point (55%
recall) in the first pass. Also adds a velocity channel (in addition to
position) and label-preserving train-time augmentation (left-right mirror,
small coordinate jitter, random temporal crop) -- THETIS only has 55
subjects, so augmentation matters more here than it would on a larger set.

Splits by subject id, not by clip, since clips from the same subject/action
are highly correlated (same person, same gym, same camera) -- a clip-level
split would leak and overstate accuracy.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Pure I/O/splitting logic lives in skeleton_records.py (no torch dependency,
# so callers like build_expert_templates.py can use it without needing torch
# installed) -- re-exported here so existing `from stroke_dataset import
# load_records, subject_disjoint_split, STROKE_CLASSES, ...` callers are unaffected.
from skeleton_records import (
    CATEGORY_TO_STROKE, STROKE_CLASSES, STROKE_TO_IDX,
    load_records, subject_disjoint_split, subject_kfold_split,
)

FIXED_T = 64  # resample every clip to this many frames

# COCO-17 left/right joint swap for mirror augmentation (nose stays put)
MIRROR_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def resample_time(kpts, target_t):
    """Linearly resample (T, V, C) along the time axis to (target_t, V, C)."""
    T = kpts.shape[0]
    if T == target_t:
        return kpts
    src_idx = np.linspace(0, T - 1, num=T)
    dst_idx = np.linspace(0, T - 1, num=target_t)
    V, C = kpts.shape[1], kpts.shape[2]
    out = np.empty((target_t, V, C), dtype=np.float32)
    for v in range(V):
        for c in range(C):
            out[:, v, c] = np.interp(dst_idx, src_idx, kpts[:, v, c])
    return out


def add_velocity(kpts):
    """kpts: (T, V, 2) -> (T, V, 4) with velocity (finite difference, first frame = 0) appended."""
    vel = np.zeros_like(kpts)
    vel[1:] = kpts[1:] - kpts[:-1]
    return np.concatenate([kpts, vel], axis=-1)


def mirror(kpts):
    """Left-right flip: negate x, swap left/right joint indices. Label-preserving --
    a mirrored forehand is still a forehand (same swing pattern, opposite-handed player)."""
    out = kpts[:, MIRROR_IDX, :].copy()
    out[..., 0] *= -1
    return out


def random_temporal_crop(kpts, min_frac=0.85):
    """Crop a random contiguous sub-window (>= min_frac of the clip) before resampling,
    a cheap way to simulate swing-boundary/timing variation."""
    T = kpts.shape[0]
    if T < 8:
        return kpts
    frac = np.random.uniform(min_frac, 1.0)
    crop_len = max(8, int(T * frac))
    start = np.random.randint(0, T - crop_len + 1)
    return kpts[start:start + crop_len]


class StrokeDataset(Dataset):
    def __init__(self, records, fixed_t=FIXED_T, augment=False, jitter_std=0.02):
        """records: list of dicts with keys kpts (T,V,2) ndarray, label, subject_id, skill_level."""
        self.records = records
        self.fixed_t = fixed_t
        self.augment = augment
        self.jitter_std = jitter_std

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        kpts = rec["kpts"]  # (T, V, 2)

        if self.augment:
            kpts = random_temporal_crop(kpts)
            if np.random.rand() < 0.5:
                kpts = mirror(kpts)

        kpts = resample_time(kpts, self.fixed_t)  # (T, V, 2)

        if self.augment:
            kpts = kpts + np.random.normal(0, self.jitter_std, kpts.shape).astype(np.float32)

        kpts = add_velocity(kpts)  # (T, V, 4)
        kpts = torch.from_numpy(kpts.astype(np.float32)).permute(2, 0, 1).contiguous()  # (C=4, T, V)
        return kpts, rec["label"], rec["subject_id"], rec["skill_level"]
