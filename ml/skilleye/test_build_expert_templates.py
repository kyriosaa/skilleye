import numpy as np
import pytest

from build_expert_templates import clip_joint_vector, compute_covariance_templates
from quality.correlation import (
    JOINT_ORDER,
    fit_covariance,
    shrink_correlation,
    shrinkage_intensity,
    shrinkage_target,
)
from quality.keypoints import (
    L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,
    L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
)
from quality.phases import PHASES, split_phases


def synthetic_clip(left_elbow_rad, trunk_rotation_rad, knee_bend_rad=0.0,
                    right_elbow_rad=np.pi / 2, T=20):
    """A clip whose left-elbow, right-elbow, trunk-rotation, and knee angles
    are the given constants for every frame, with a clear mid-clip
    right-wrist speed peak (same trick as quality/test_score.py's
    make_straight_arm_clip -- radial motion along a fixed direction from the
    elbow, so the elbow angle is invariant to the ramp's magnitude) so
    split_phases finds a contact frame away from either boundary and every
    phase gets a nonzero frame count. All 5 tracked joints must vary across a
    clip set for a realistic (nonsingular) covariance -- any one held at a
    fixed default makes that dimension identical, and thus zero-variance,
    across every clip."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)

    kpts[:, L_SHOULDER] = (0, 2)
    kpts[:, L_ELBOW] = (0, 1)
    # v1 = L_SHOULDER - L_ELBOW = (0, 1); place L_WRIST so the angle at
    # L_ELBOW between v1 and v2 = L_WRIST - L_ELBOW is left_elbow_rad.
    offset = (np.sin(left_elbow_rad), np.cos(left_elbow_rad))
    kpts[:, L_WRIST] = (0 + offset[0], 1 + offset[1])

    kpts[:, R_SHOULDER] = (1, 2)
    kpts[:, R_ELBOW] = (1, 1)
    # v1 = R_SHOULDER - R_ELBOW = (0, 1); ramp the wrist radially along the
    # fixed direction at angle right_elbow_rad from v1, so the angle at
    # R_ELBOW is right_elbow_rad regardless of the ramp's magnitude.
    direction = (np.sin(right_elbow_rad), np.cos(right_elbow_rad))
    magnitude = 0.0
    for t in range(T):
        step = 1.0 if t < T // 2 - 1 else (5.0 if t == T // 2 - 1 else 0.2)
        magnitude += step
        kpts[t, R_WRIST] = (1 + magnitude * direction[0], 1 + magnitude * direction[1])

    kpts[:, L_HIP] = (0, -1)
    # hip_vec = R_HIP - L_HIP; rotate it by trunk_rotation_rad from (1, 0).
    kpts[:, R_HIP] = (
        0 + np.cos(trunk_rotation_rad),
        -1 + np.sin(trunk_rotation_rad),
    )
    # v1 = L_HIP - L_KNEE = (0, 1) at L_KNEE=(0,-2); place L_ANKLE so the
    # angle there is knee_bend_rad (same construction as the elbow above,
    # mirrored downward). Both knees get the same bend for simplicity.
    kpts[:, L_KNEE] = (0, -2)
    knee_offset = (np.sin(knee_bend_rad), -np.cos(knee_bend_rad))
    kpts[:, L_ANKLE] = (0 + knee_offset[0], -2 + knee_offset[1])
    kpts[:, R_KNEE] = (1, -2)
    kpts[:, R_ANKLE] = (1 + knee_offset[0], -2 + knee_offset[1])
    return kpts


def expected_vectors_for_phase(clips, phase):
    """Ground truth, computed the same way production code does (via
    clip_joint_vector) -- this test is about compute_covariance_templates's
    aggregation/grouping, not re-deriving quality/angles.py's math."""
    vectors = []
    for kpts in clips:
        vec = clip_joint_vector(split_phases(kpts)[phase])
        if vec is not None:
            vectors.append(vec)
    return vectors


def test_clip_joint_vector_matches_joint_order():
    kpts = synthetic_clip(left_elbow_rad=np.pi / 2, trunk_rotation_rad=0.3)
    vec = clip_joint_vector(kpts)
    assert vec.shape == (len(JOINT_ORDER),)
    assert vec[JOINT_ORDER.index("left_elbow")] == pytest.approx(np.pi / 2, abs=1e-5)


def test_clip_joint_vector_returns_none_for_empty_phase():
    empty_phase = np.zeros((0, 17, 2), dtype=np.float32)
    assert clip_joint_vector(empty_phase) is None


def test_compute_covariance_templates_matches_direct_fit_and_shrink():
    backhand_clips = [
        synthetic_clip(left_elbow_rad=angle, trunk_rotation_rad=trunk,
                        knee_bend_rad=knee, right_elbow_rad=relbow)
        for angle, trunk, knee, relbow in [
            (2.0, 0.1, 2.6, 1.3), (2.2, 0.3, 2.4, 1.5), (1.8, 0.5, 2.9, 1.1),
            (2.5, 0.2, 2.2, 1.6), (1.6, 0.4, 2.7, 1.2),
        ]
    ]
    records = [{"stroke": "backhand", "kpts": kpts} for kpts in backhand_clips]
    records.append({"stroke": "forehand", "kpts": backhand_clips[0]})  # only 1 clip

    result = compute_covariance_templates(records, stroke_classes=["backhand", "forehand"])

    p = len(JOINT_ORDER)
    target_corr = shrinkage_target()
    for phase in PHASES:
        vectors = expected_vectors_for_phase(backhand_clips, phase)
        n = len(vectors)
        assert n >= 2, "test fixture should produce nonempty phases"
        expected_mean, expected_cov = fit_covariance(np.array(vectors))
        expected_shrunk = shrink_correlation(
            expected_cov, target_corr, shrinkage_intensity(n, p))

        entry = result["backhand"][phase]
        assert entry["joint_order"] == JOINT_ORDER
        assert entry["n"] == n
        assert np.array(entry["mean"]) == pytest.approx(expected_mean)
        assert np.array(entry["cov"]) == pytest.approx(expected_shrunk)


def test_compute_covariance_templates_skips_strokes_with_too_few_clips():
    records = [{"stroke": "forehand", "kpts": synthetic_clip(2.0, 0.1)}]  # n=1
    result = compute_covariance_templates(records, stroke_classes=["forehand"])
    assert result["forehand"] == {}
