"""
Correlated z-scoring for quality scoring (Module A): instead of comparing each
joint's phase-mean angle to its own independent (mean, std), compare it to its
*conditional* mean/variance given the other joints' values in the same phase,
via a per-(stroke, phase) covariance matrix. Design:
docs/superpowers/specs/2026-08-14-correlated-zscore-module-a-design.md
"""
import numpy as np

JOINT_ORDER = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
               "left_knee", "right_knee", "trunk_rotation"]


def fit_covariance(vectors):
    """vectors: (N, p) array-like of per-clip joint vectors. Returns (mean,
    cov): mean is (p,), cov is (p, p), population covariance (ddof=0, matching
    build_expert_templates.py's existing arr.std() convention)."""
    arr = np.asarray(vectors, dtype=np.float64)
    mean = arr.mean(axis=0)
    cov = np.cov(arr.T, bias=True)
    return mean, cov


def conditional_zscore(values, mean, cov):
    """values, mean: (p,). cov: (p, p). For each joint i, z-scores values[i]
    against its conditional mean/variance given every other joint's observed
    value (the multivariate-Gaussian conditional, via cov's Schur
    complement) -- not against its own marginal (mean, std). Falls back to
    the marginal z-score for joint i if the other joints' (p-1, p-1)
    submatrix isn't invertible (degenerate data)."""
    values = np.asarray(values, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    p = values.shape[0]
    z = np.empty(p, dtype=np.float64)

    for i in range(p):
        others = [j for j in range(p) if j != i]
        cov_ii = cov[i, i]
        cov_i_others = cov[i, others]
        cov_others = cov[np.ix_(others, others)]
        deviation_others = values[others] - mean[others]

        try:
            # cov_i_others @ inv(cov_others) @ deviation_others, without
            # forming the inverse explicitly.
            solved = np.linalg.solve(cov_others, deviation_others)
            cond_mean = mean[i] + cov_i_others @ solved
            cond_var = cov_ii - cov_i_others @ np.linalg.solve(cov_others, cov_i_others)
        except np.linalg.LinAlgError:
            cond_mean, cond_var = None, None

        if cond_mean is None or cond_var <= 1e-9:
            # Degenerate submatrix (or a conditional variance that collapsed
            # to ~0) -- fall back to the marginal z-score for this joint
            # rather than dividing by near-zero or propagating a NaN.
            z[i] = (values[i] - mean[i]) / np.sqrt(cov_ii)
        else:
            z[i] = (values[i] - cond_mean) / np.sqrt(cond_var)

    return z


# Elliott (2006)'s kinetic-chain contribution figures (shoulder 10-15%,
# upper-arm internal rotation 40%, wrist 20-30%) describe % contribution to
# racket-head speed, not a correlation coefficient -- these priors are a
# small, clearly-labeled nudge in the direction Elliott's chain describes
# (trunk rotation driving the arm, adjacent segments moving together), not a
# citation of specific numbers from the paper. "Upper-arm internal rotation"
# and "wrist" still aren't separately tracked (see quality/angles.py's note
# on left_shoulder/right_shoulder -- internal rotation isn't measurable from
# a 2D camera at all); left_shoulder/right_shoulder is the nearest tracked
# proxy for Elliott's "shoulder" contributor.
UPPER_LIMB_TRUNK_PRIOR_CORRELATION = 0.3
SHOULDER_ELBOW_PRIOR_CORRELATION = 0.3  # adjacent same-arm segments


def shrinkage_target():
    """Returns a (p, p) prior correlation matrix over JOINT_ORDER: identity,
    except a modest positive prior (a) between each shoulder/elbow and trunk
    rotation (see UPPER_LIMB_TRUNK_PRIOR_CORRELATION) and (b) between each
    arm's shoulder and elbow (see SHOULDER_ELBOW_PRIOR_CORRELATION). Used to
    regularize the empirical correlation matrix for the thin (n=57)
    volley/smash classes."""
    p = len(JOINT_ORDER)
    target = np.eye(p)

    def couple(name_a, name_b, value):
        i, j = JOINT_ORDER.index(name_a), JOINT_ORDER.index(name_b)
        target[i, j] = value
        target[j, i] = value

    for side in ("left", "right"):
        couple(f"{side}_shoulder", "trunk_rotation", UPPER_LIMB_TRUNK_PRIOR_CORRELATION)
        couple(f"{side}_elbow", "trunk_rotation", UPPER_LIMB_TRUNK_PRIOR_CORRELATION)
        couple(f"{side}_shoulder", f"{side}_elbow", SHOULDER_ELBOW_PRIOR_CORRELATION)
    return target


def shrinkage_intensity(n, p):
    """Simple n-scaled shrinkage heuristic: more samples -> less reliance on
    the prior. Not a formally fitted estimator (e.g. Ledoit-Wolf) -- a sane
    default in the same spirit as this project's other not-yet-calibrated
    scoring constants (FLAG_THRESHOLD, SCORE_SCALE in quality/score.py)."""
    return float(np.clip(p / max(n - p, 1), 0.0, 0.6))


def shrink_correlation(cov, target_corr, intensity):
    """cov: (p, p) empirical covariance. target_corr: (p, p) prior correlation
    matrix (e.g. from shrinkage_target()). intensity: 0 = pure empirical, 1 =
    pure target. Returns a (p, p) covariance: the *correlation* structure is
    blended with the target, then rescaled back using cov's own (unshrunk)
    variances -- variances always come from the data."""
    cov = np.asarray(cov, dtype=np.float64)
    std = np.sqrt(np.diag(cov))
    # A ~zero-variance joint's covariance with every other joint is also
    # ~zero (|cov_ij| <= std_i * std_j), so its row/col of empirical_corr is
    # mathematically 0/0 -- substitute a safe denominator only to avoid the
    # NaN; the final rescale below multiplies back by the *true* std (still
    # ~0), so the degenerate joint's output row/col is exactly zero either way.
    safe_std = np.where(std > 1e-9, std, 1.0)
    inv_std = np.diag(1.0 / safe_std)
    empirical_corr = inv_std @ cov @ inv_std

    shrunk_corr = (1.0 - intensity) * empirical_corr + intensity * target_corr

    std_outer = np.diag(std)
    return std_outer @ shrunk_corr @ std_outer
