import numpy as np
import pytest

from quality.correlation import (
    JOINT_ORDER,
    conditional_zscore,
    fit_covariance,
    shrink_correlation,
    shrinkage_intensity,
    shrinkage_target,
)


def test_fit_covariance_matches_hand_computed_mean_and_cov():
    vectors = np.array([[1.0, 2.0], [3.0, 1.0], [5.0, 6.0]])
    mean, cov = fit_covariance(vectors)
    assert mean == pytest.approx([3.0, 3.0])
    assert cov == pytest.approx(np.array([[8 / 3, 8 / 3], [8 / 3, 14 / 3]]))


def test_conditional_zscore_matches_hand_computed_bivariate_case():
    # Same mean/cov as the fit_covariance test above, computed independently
    # by hand (see docs/superpowers/specs/2026-08-14-correlated-zscore-module-a-design.md):
    # conditioning joint 0 on joint 1's observed value of 9 gives a
    # conditional mean of 45/7 and conditional variance of 8/7.
    mean = np.array([3.0, 3.0])
    cov = np.array([[8 / 3, 8 / 3], [8 / 3, 14 / 3]])
    values = np.array([8.0, 9.0])

    z = conditional_zscore(values, mean, cov)

    assert z[0] == pytest.approx(1.469936830518334)


def test_conditional_zscore_matches_marginal_when_joints_are_uncorrelated():
    # Diagonal covariance -- no coupling between joints -- so conditioning on
    # the other joints should change nothing: this must reduce to the
    # existing independent z-score, (value - mean) / std.
    mean = np.array([3.0, 3.0, 3.0])
    cov = np.diag([2.0, 5.0, 9.0])
    values = np.array([5.0, 1.0, 6.0])

    z = conditional_zscore(values, mean, cov)

    expected = (values - mean) / np.sqrt(np.diag(cov))
    assert z == pytest.approx(expected)


def test_conditional_zscore_falls_back_to_marginal_when_submatrix_singular():
    # Joints 1 and 2 are perfectly correlated (identical), so the (1, 2)
    # submatrix used to condition joint 0 is singular. Joint 0 should fall
    # back to its own marginal z-score instead of raising or returning NaN.
    mean = np.array([3.0, 3.0, 3.0])
    cov = np.array([
        [4.0, 0.0, 0.0],
        [0.0, 2.0, 2.0],
        [0.0, 2.0, 2.0],
    ])
    values = np.array([7.0, 1.0, 1.0])

    z = conditional_zscore(values, mean, cov)

    assert z[0] == pytest.approx((7.0 - 3.0) / np.sqrt(4.0))


def test_shrinkage_target_has_prior_coupling_between_elbows_and_trunk_rotation():
    target = shrinkage_target()
    trunk = JOINT_ORDER.index("trunk_rotation")
    left_elbow = JOINT_ORDER.index("left_elbow")
    right_elbow = JOINT_ORDER.index("right_elbow")
    left_knee = JOINT_ORDER.index("left_knee")

    assert target.shape == (len(JOINT_ORDER), len(JOINT_ORDER))
    assert target == pytest.approx(target.T)  # a correlation matrix is symmetric
    assert np.diag(target) == pytest.approx(np.ones(len(JOINT_ORDER)))
    # The one coupling Elliott (2006) maps onto joints this project tracks.
    assert target[left_elbow, trunk] > 0
    assert target[right_elbow, trunk] > 0
    # Knees are a separate force-generation subsystem -- no prior coupling.
    assert target[left_elbow, left_knee] == 0
    assert target[left_knee, trunk] == 0


def test_shrink_correlation_returns_original_covariance_at_zero_intensity():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    target_corr = np.array([[1.0, 0.8], [0.8, 1.0]])

    shrunk = shrink_correlation(cov, target_corr, intensity=0.0)

    assert shrunk == pytest.approx(cov)


def test_shrink_correlation_returns_target_correlation_at_full_intensity():
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    target_corr = np.array([[1.0, 0.8], [0.8, 1.0]])

    shrunk = shrink_correlation(cov, target_corr, intensity=1.0)

    # Variances (diagonal) come from the data and are never shrunk; only the
    # off-diagonal coupling moves fully to the target correlation (0.8 * 2 * 3).
    assert np.diag(shrunk) == pytest.approx([4.0, 9.0])
    assert shrunk[0, 1] == pytest.approx(0.8 * 2.0 * 3.0)


def test_shrink_correlation_handles_a_zero_variance_joint_without_nan():
    # Joint 0 has exactly zero variance (e.g. a degenerate/locked measurement)
    # -- its covariance with every other joint must also be 0 (|cov_ij| <=
    # std_i*std_j), so shrinking must not touch it, and must not produce NaN
    # via a division by its zero std (score.py's existing std<=1e-6 guard is
    # the same concern for the independent-z-score path).
    cov = np.array([[0.0, 0.0], [0.0, 9.0]])
    target_corr = np.array([[1.0, 0.9], [0.9, 1.0]])

    shrunk = shrink_correlation(cov, target_corr, intensity=0.5)

    assert not np.any(np.isnan(shrunk))
    assert shrunk[0, 0] == pytest.approx(0.0)
    assert shrunk[0, 1] == pytest.approx(0.0)
    assert shrunk[1, 1] == pytest.approx(9.0)


def test_shrinkage_intensity_is_lower_for_the_larger_expert_template_classes():
    # Real n's from ml/results/quality_templates/templates.json: 171 clips for
    # backhand/forehand/serve, 57 for the volleys/smash -- the thinner classes
    # should lean on the prior more.
    p = len(JOINT_ORDER)
    intensity_large = shrinkage_intensity(n=171, p=p)
    intensity_small = shrinkage_intensity(n=57, p=p)

    assert 0.0 < intensity_large < intensity_small < 0.6
