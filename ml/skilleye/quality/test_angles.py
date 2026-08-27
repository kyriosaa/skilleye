import numpy as np
import pytest

from quality.angles import (
    joint_angle_series, trunk_rotation_series, compute_all_angles,
    phase_mean_angles, JOINT_DEFINITIONS,
    signed_shoulder_pelvis_twist_series, signed_pelvic_rotation_series,
)
from quality.keypoints import L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, L_HIP, R_HIP


def test_joint_angle_series_right_angle():
    # shoulder at (0,1), elbow at (0,0), wrist at (1,0) -> 90 degrees at the elbow
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 1)
    kpts[0, L_ELBOW] = (0, 0)
    kpts[0, L_WRIST] = (1, 0)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert result.shape == (1,)
    assert np.isclose(result[0], np.pi / 2, atol=1e-5)


def test_joint_angle_series_straight_arm_is_pi():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 1)
    kpts[0, L_ELBOW] = (0, 0)
    kpts[0, L_WRIST] = (0, -1)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert np.isclose(result[0], np.pi, atol=1e-5)


def test_joint_angle_series_empty_input():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = joint_angle_series(kpts, L_SHOULDER, L_ELBOW, L_WRIST)
    assert result.shape == (0,)


def test_trunk_rotation_series_parallel_lines_is_zero():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_SHOULDER] = (0, 0)
    kpts[0, R_SHOULDER] = (1, 0)
    kpts[0, L_HIP] = (0, -1)
    kpts[0, R_HIP] = (1, -1)
    result = trunk_rotation_series(kpts)
    assert np.isclose(result[0], 0.0, atol=1e-5)


def test_compute_all_angles_has_expected_keys_and_shapes():
    kpts = np.zeros((3, 17, 2), dtype=np.float32)
    result = compute_all_angles(kpts)
    assert set(result.keys()) == set(JOINT_DEFINITIONS.keys()) | {"trunk_rotation"}
    for series in result.values():
        assert series.shape == (3,)


def test_left_shoulder_is_a_right_angle_when_upper_arm_is_perpendicular_to_torso():
    # hip->shoulder points straight up; shoulder->elbow points sideways -> 90 deg
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_HIP] = (0, 0)
    kpts[0, L_SHOULDER] = (0, 1)
    kpts[0, L_ELBOW] = (1, 1)
    result = joint_angle_series(kpts, *JOINT_DEFINITIONS["left_shoulder"])
    assert np.isclose(result[0], np.pi / 2, atol=1e-5)


def test_right_shoulder_is_pi_when_arm_hangs_straight_down_in_line_with_torso():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, R_HIP] = (0, 0)
    kpts[0, R_SHOULDER] = (0, 1)
    kpts[0, R_ELBOW] = (0, 2)
    result = joint_angle_series(kpts, *JOINT_DEFINITIONS["right_shoulder"])
    assert np.isclose(result[0], np.pi, atol=1e-5)


def test_phase_mean_angles_returns_none_for_empty_phase():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = phase_mean_angles(kpts)
    assert all(v is None for v in result.values())


def test_phase_mean_angles_returns_float_for_nonempty_phase():
    kpts = np.zeros((3, 17, 2), dtype=np.float32)
    kpts[:, L_SHOULDER] = (0, 1)
    kpts[:, L_ELBOW] = (0, 0)
    kpts[:, L_WRIST] = (1, 0)
    result = phase_mean_angles(kpts)
    assert isinstance(result["left_elbow"], float)
    assert np.isclose(result["left_elbow"], np.pi / 2, atol=1e-5)


# --- signed angles (Module B: quality/skill_rules.py needs sign, which the
# arccos-based trunk_rotation_series above can't give) ---

def test_signed_shoulder_pelvis_twist_positive_when_shoulders_rotated_ccw_from_hips():
    theta = np.radians(30)
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_HIP] = (-1, 0)
    kpts[0, R_HIP] = (1, 0)  # hip line horizontal
    kpts[0, L_SHOULDER] = (-np.cos(theta), -np.sin(theta))
    kpts[0, R_SHOULDER] = (np.cos(theta), np.sin(theta))  # shoulder line at +theta
    result = signed_shoulder_pelvis_twist_series(kpts)
    assert result[0] == pytest.approx(theta, abs=1e-5)


def test_signed_shoulder_pelvis_twist_negative_when_shoulders_rotated_cw_from_hips():
    theta = np.radians(30)
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_HIP] = (-1, 0)
    kpts[0, R_HIP] = (1, 0)
    kpts[0, L_SHOULDER] = (-np.cos(theta), np.sin(theta))
    kpts[0, R_SHOULDER] = (np.cos(theta), -np.sin(theta))  # shoulder line at -theta
    result = signed_shoulder_pelvis_twist_series(kpts)
    assert result[0] == pytest.approx(-theta, abs=1e-5)


def test_signed_shoulder_pelvis_twist_zero_when_parallel():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_HIP], kpts[0, R_HIP] = (-1, 0), (1, 0)
    kpts[0, L_SHOULDER], kpts[0, R_SHOULDER] = (-1, 1), (1, 1)  # parallel, shifted up
    result = signed_shoulder_pelvis_twist_series(kpts)
    assert result[0] == pytest.approx(0.0, abs=1e-5)


def test_signed_shoulder_pelvis_twist_empty_input():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = signed_shoulder_pelvis_twist_series(kpts)
    assert result.shape == (0,)


def test_signed_pelvic_rotation_zero_when_hip_line_horizontal():
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_HIP], kpts[0, R_HIP] = (-1, 0), (1, 0)
    result = signed_pelvic_rotation_series(kpts)
    assert result[0] == pytest.approx(0.0, abs=1e-5)


def test_signed_pelvic_rotation_matches_hip_line_tilt():
    theta = np.radians(20)
    kpts = np.zeros((1, 17, 2), dtype=np.float32)
    kpts[0, L_HIP] = (-np.cos(theta), -np.sin(theta))
    kpts[0, R_HIP] = (np.cos(theta), np.sin(theta))
    result = signed_pelvic_rotation_series(kpts)
    assert result[0] == pytest.approx(theta, abs=1e-5)


def test_signed_pelvic_rotation_empty_input():
    kpts = np.zeros((0, 17, 2), dtype=np.float32)
    result = signed_pelvic_rotation_series(kpts)
    assert result.shape == (0,)
