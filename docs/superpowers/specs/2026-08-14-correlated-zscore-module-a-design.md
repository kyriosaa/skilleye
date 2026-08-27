# Correlated Z-Score for Quality Scoring — Design (Module A)

Date: 2026-08-14
Status: Approved for implementation

## Problem

A teammate proposed two extensions to the rule-based quality scorer (`quality/score.py`),
handed over as two documents (`updated.pdf`, `Module_A_B_Reading_Plan.docx`). This spec
covers **Module A** only — Module B (skill-level-specific rules for backhand volley, from
Katsumi et al. and Aydin & Aydemir) is a separate, later spec.

Current scoring (`score_clip` in `quality/score.py`) treats each of the 5 tracked joint
angles (left/right elbow, left/right knee, trunk rotation) as independent: each phase-mean
angle gets its own z-score against that joint's own (mean, std) in `templates.json`, with no
knowledge of what the other 4 joints are doing in the same phase. This has two documented
consequences:

1. **False positives from natural coupling.** A joint that deviates only because it is
   biomechanically "pulled along" by another joint (e.g. trunk rotation correlating with
   elbow extension through the kinetic chain) gets flagged on its own merits, with no way to
   express "this is expected given the rest of the swing."
2. **No stroke-specific coupling structure.** Power strokes (serve, groundstrokes) and
   precision strokes (volleys) are known (Elliott 2006; Knudson & Elliott 2004) to route
   force through the kinetic chain differently, but independent z-scoring can't represent
   that — every joint is scored in isolation regardless of stroke.

## Approach: conditional z-score from a per-(stroke, phase) covariance matrix

For each (stroke, phase), estimate a 5×5 covariance matrix over the joint vector
`[left_elbow, right_elbow, left_knee, right_knee, trunk_rotation]` from the same expert
clips already used to build `templates.json` (171 clips for backhand/forehand/serve, 57 for
the volleys and smash — verified against the current `templates.json`, not the "24 clips"
figure in the handed-over documents, which referred to something else).

Instead of comparing a query clip's joint value to the template's unconditional
`(mean, std)`, compute its **conditional** mean and variance given the other 4 joints'
observed values in the same phase (a standard multivariate-Gaussian conditional, via the
covariance matrix's Schur complement), then z-score against that:

```
z_i = (x_i - E[x_i | x_{-i}]) / sqrt(Var[x_i | x_{-i}])
```

This directly answers "is joint i off, *given* what the rest of the body is doing" instead
of "is joint i off, on its own" — addressing limitation #1. Because the covariance matrix is
estimated separately per stroke class, the coupling structure is already stroke-specific
(limitation #2) with no extra mechanism needed.

**Output format is unchanged**: still 5 joints × 3 phases = 15 z-scored deviations, same
shape as today. `score.py`'s table/suggestion/overall-score logic downstream of the z-score
does not need to change — only how each `z` is computed.

### Handling the small-n classes (volleys, smash: n=57)

A 5×5 covariance from 57 samples is usable but noisier than from 171. Rather than pull in
Elliott's kinetic-chain contribution percentages as literal correlation values (they
describe a different physical quantity — % contribution to racket-head speed, not a
correlation coefficient between two angle signals — and cite shoulder/wrist joints this
project doesn't track separately, only a combined elbow-flexion angle), they're used only to
build a small **shrinkage target**: a structured correlation matrix with a modest positive
prior between each arm's elbow angle and trunk rotation (the one coupling in Elliott's
kinetic-chain description that maps onto joints this project actually measures), and near-zero
prior elsewhere (knees are a separate force-generation subsystem, not part of Elliott's
upper-limb chain figures). The empirical correlation matrix is shrunk toward this target with
an intensity that scales down as `n` grows, so the large classes (n=171) stay close to pure
empirical estimates and only the thin classes lean on the prior. This mirrors the project's
existing "sane default, checked with a smoke test, not formally calibrated" posture for
`FLAG_THRESHOLD`/`SCORE_SCALE` — not a claim of a rigorously fitted shrinkage estimator.

### Fallback

If a phase's covariance submatrix is singular or near-singular (degenerate data), conditional
z-score falls back to the existing independent z-score for that (phase, joint) rather than
producing NaN/garbage — consistent with `score.py`'s existing pattern of surfacing missing
template coverage (`"insufficient template data"`) rather than hiding it.

## Components

### 1. `quality/correlation.py` (new)

- `JOINT_ORDER`: fixed 5-joint ordering shared with `angles.py`/`score.py`.
- `fit_covariance(vectors)` — mean vector + population covariance (ddof=0, matching
  `build_expert_templates.py`'s existing `arr.std()` convention) from a list of per-clip
  5-vectors.
- `shrinkage_target()` — the small hand-specified prior correlation matrix described above,
  with a comment citing Elliott (2006) and the shoulder/wrist caveat.
- `shrink_correlation(cov, target, n)` — converts `cov` to a correlation matrix, blends with
  `target` using an `n`-scaled intensity, rescales back to a covariance using the original
  (unshrunk) variances. Variances are never shrunk, only the coupling structure.
- `conditional_zscore(values, mean, cov)` — returns one z per joint, using the Schur
  complement; falls back to marginal (independent) z-score per-joint if the relevant
  submatrix isn't invertible.

### 2. `build_expert_templates.py` (extend)

Alongside the existing per-joint `templates[stroke][phase][joint] = {mean, std, n}`, also
collect each clip's full 5-vector and save a new top-level key:

```
covariance[stroke][phase] = {"joint_order": [...], "mean": [...], "cov": [[...]], "n": int}
```

Additive only — the existing `templates` key and its consumers (`score.py`, `app.py`) are
untouched, so nothing currently working can regress.

### 3. `quality/score.py` (extend)

Add `score_clip_correlated(kpts, stroke_class, templates, covariance)` alongside the existing
`score_clip`, sharing `phase_mean_angles`/`suggestion_text`/table-building shape, but calling
`conditional_zscore` instead of the independent `(value - mean) / std`. `score_clip` itself
is not modified — this is an additive v2, in keeping with this project's existing v1/v2/v3
pattern of keeping prior validated behavior available rather than overwriting it. Wiring this
into `app.py`'s UI is a follow-up step, out of scope here.

## Testing

- `quality/test_correlation.py`: `fit_covariance` against a hand-computable toy dataset;
  `conditional_zscore` against a known bivariate-Gaussian case worked out by hand (verify the
  conditional mean/variance formula, not just that it runs); `shrink_correlation` blends
  correctly at shrinkage=0 (pure empirical) and shrinkage=1 (pure target); singular-matrix
  fallback returns the same value as the existing independent z-score.
- `quality/test_score.py`: add cases for `score_clip_correlated` mirroring the existing
  `score_clip` tests (flags large deviation, doesn't flag a close match, output shape
  unchanged at 15 rows).

## Out of scope

- Module B (skill-level-specific rules) — separate spec.
- Wiring `score_clip_correlated` into `app.py`'s Streamlit UI.
- A formally fitted (e.g. Ledoit-Wolf) shrinkage estimator — the simple `n`-scaled heuristic
  is consistent with this project's existing not-yet-calibrated posture on scoring constants.
