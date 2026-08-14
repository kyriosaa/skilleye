import numpy as np
import pytest

from quality.keypoints import L_HIP, R_HIP, L_SHOULDER, R_SHOULDER, R_WRIST
from quality.phases import CONTACT_WINDOW, detect_contact_frame
from quality.skill_rules import (
    check_excessive_pelvic_rotation,
    check_shoulder_pelvis_twist_reversal,
    check_volley_swing_effort,
    evaluate_backhand_volley_skill_rules,
    g_to_mps2,
)


def _hip_shoulder_dirs(pelvic_deg, twist_deg):
    pelvic = np.radians(pelvic_deg)
    shoulder_angle = pelvic + np.radians(twist_deg)
    hip_dir = np.array([np.cos(pelvic), np.sin(pelvic)], dtype=np.float32)
    shoulder_dir = np.array([np.cos(shoulder_angle), np.sin(shoulder_angle)], dtype=np.float32)
    return hip_dir, shoulder_dir


def make_backhand_volley_clip(backswing_pelvic_deg, backswing_twist_deg,
                               contact_pelvic_deg, contact_twist_deg, T=20):
    """A clip whose hip/shoulder geometry gives one (pelvic_rotation, twist)
    pair during backswing and a different pair from contact onward -- the
    real detect_contact_frame (same wrist-speed-peak trick as
    quality/test_score.py's make_straight_arm_clip) decides where that
    switch happens, so this fixture drives the same code path
    evaluate_backhand_volley_skill_rules does, not a hand-guessed frame index."""
    kpts = np.zeros((T, 17, 2), dtype=np.float32)

    x = 0.0
    for t in range(T):
        step = 1.0 if t < T // 2 - 1 else (5.0 if t == T // 2 - 1 else 0.2)
        x += step
        kpts[t, R_WRIST] = (1 + x, 1)

    contact = detect_contact_frame(kpts)
    contact_start = max(0, contact - CONTACT_WINDOW)

    backswing_hip, backswing_shoulder = _hip_shoulder_dirs(backswing_pelvic_deg, backswing_twist_deg)
    contact_hip, contact_shoulder = _hip_shoulder_dirs(contact_pelvic_deg, contact_twist_deg)
    shoulder_offset = np.array([0, 2], dtype=np.float32)

    for t in range(T):
        hip_dir, shoulder_dir = (
            (backswing_hip, backswing_shoulder) if t < contact_start
            else (contact_hip, contact_shoulder)
        )
        kpts[t, L_HIP] = -hip_dir
        kpts[t, R_HIP] = hip_dir
        kpts[t, L_SHOULDER] = -shoulder_dir + shoulder_offset
        kpts[t, R_SHOULDER] = shoulder_dir + shoulder_offset

    return kpts


def test_g_to_mps2_matches_standard_gravity():
    assert g_to_mps2(1.0) == pytest.approx(9.80665)
    assert g_to_mps2(0.0) == pytest.approx(0.0)


class TestTwistReversal:

    def test_flags_when_sign_differs_between_phases(self):
        # Matches Katsumi et al. (2026)'s less-skilled pattern: near-zero/
        # negative at backswing, positive at contact.
        assert check_shoulder_pelvis_twist_reversal(
            twist_backswing_deg=-7.7, twist_contact_deg=10.1) is True

    def test_does_not_flag_when_sign_is_consistent(self):
        # Matches the skilled pattern: negative at both phases.
        assert check_shoulder_pelvis_twist_reversal(
            twist_backswing_deg=-18.9, twist_contact_deg=-15.5) is False


class TestExcessivePelvicRotation:

    def test_flags_both_phases_beyond_the_less_skilled_side(self):
        # Deep in less-skilled territory: -46.7 (backswing), -65.7 (contact)
        # are literally the less-skilled group's own means.
        result = check_excessive_pelvic_rotation(
            pelvic_backswing_deg=-46.7, pelvic_contact_deg=-65.7)
        assert result == {"backswing": True, "contact": True}

    def test_does_not_flag_values_at_the_skilled_group_means(self):
        result = check_excessive_pelvic_rotation(
            pelvic_backswing_deg=-29.6, pelvic_contact_deg=-37.1)
        assert result == {"backswing": False, "contact": False}


class TestVolleySwingEffort:

    def test_flags_high_acceleration_near_the_amateur_mean(self):
        result = check_volley_swing_effort(peak_accel_mps2=57.09)
        assert result["flagged"] is True
        assert result["note"] is not None

    def test_does_not_flag_low_acceleration_near_the_elite_mean(self):
        result = check_volley_swing_effort(peak_accel_mps2=48.12)
        assert result["flagged"] is False
        assert result["note"] is None


class TestEvaluateBackhandVolleySkillRules:

    def test_flags_the_less_skilled_pattern_end_to_end(self):
        # The less-skilled group's own reference means, from both phases.
        kpts = make_backhand_volley_clip(
            backswing_pelvic_deg=-46.7, backswing_twist_deg=-7.7,
            contact_pelvic_deg=-65.7, contact_twist_deg=10.1,
        )
        result = evaluate_backhand_volley_skill_rules(kpts)
        rule_names = {f["rule"] for f in result["flags"]}
        assert "shoulder_pelvis_twist_reversal" in rule_names
        assert "excessive_pelvic_rotation" in rule_names
        pelvic_phases = {f["phase"] for f in result["flags"]
                         if f["rule"] == "excessive_pelvic_rotation"}
        assert pelvic_phases == {"backswing", "contact"}

    def test_does_not_flag_the_skilled_pattern_end_to_end(self):
        kpts = make_backhand_volley_clip(
            backswing_pelvic_deg=-29.6, backswing_twist_deg=-18.9,
            contact_pelvic_deg=-37.1, contact_twist_deg=-15.5,
        )
        result = evaluate_backhand_volley_skill_rules(kpts)
        assert result["flags"] == []
