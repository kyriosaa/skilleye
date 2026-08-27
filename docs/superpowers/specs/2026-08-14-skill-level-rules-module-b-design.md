# Skill-Level-Specific Error Detection Rules — Design (Module B)

Date: 2026-08-14
Status: Approved for implementation

## Problem

Module A (`docs/superpowers/specs/2026-08-14-correlated-zscore-module-a-design.md`) addressed
limitations #1 and #3 of the independent z-score scorer. This spec covers **Module B**:
limitation #2 -- "cannot detect errors with subtle structural patterns (e.g., the relationship
between two joints reversing sign across two time points)". Module B is a set of small,
hand-crafted rules, each encoding one specific finding from a paper that directly compares
skilled vs. less-skilled players (independent z-scoring against expert-only templates can't
express this -- it has no "less-skilled" reference to compare against at all).

### Sourcing note

The exact reference numbers below were extracted from an automated proxy-read of each paper's
published page (not a manual read of the PDF). They're quoted verbatim with citations so
they're easy to spot-check against the source if precision matters later.

### What ended up implementable

Two papers were confirmed and yielded real, specific, rule-shaped findings:

- **Katsumi, K., Koda, H., & Kida, N. (2026).** Analysis of Upper-Limb Movement
  Characteristics in Tennis Volleys Based on Skill-Level Differences: Kinematic Features of
  the Backhand Versus Forehand Volley. *Journal of Functional Morphology and Kinesiology*,
  11(2), 203. -- backhand-volley-only (the paper itself reports skill-related differences are
  more pronounced in BV than FV); gives specific (mean, SD) pelvic-rotation and
  shoulder-pelvis-twist angles per skill group per phase.
- **Aydin, E.H., & Aydemir, O. (2026).** A Robust Deep Learning Framework for Skill Level
  Discrimination in Tennis Strokes Using Bilateral IMU Measurements. *Sensors*, 26(10), 3273.
  -- gives peak dominant-hand acceleration (mean, SD) by skill level for the volley.

A third candidate -- "Kinematic and Muscle Activation Differences Between High-Performance and
Intermediate Tennis Players During the Forehand Drive" (*Sensors*, 26(7), 2244) -- was read but
turned out **not implementable** here: its findings are reported via statistical parametric
mapping over phase-normalized curves (no simple group means/thresholds) and its EMG results
need surface-electrode hardware this project doesn't have. Not attempted; flagged for a future
pass only if a differently-shaped finding from it turns up.

## Approach

### Rule 1 & 2: Katsumi (2026) backhand-volley rules -- runs on the existing skeleton pipeline today

Both rules need a **signed** angle, which the existing `trunk_rotation_series`
(`quality/angles.py`) can't give -- it's built on `arccos`, which only returns an unsigned
`[0, pi]` magnitude and can't represent the sign-reversal Katsumi describes. Two new signed
series functions are added to `angles.py` (its docstring already invites this: "a new series
function, for trunk_rotation-style pairs -- nothing else needs to change"), using
`arctan2(cross_z, dot)` of the relevant vectors instead of `arccos`:

- **`signed_shoulder_pelvis_twist_series`** -- signed angle from the hip line to the shoulder
  line.
- **`signed_pelvic_rotation_series`** -- signed angle of the hip line relative to horizontal
  (an image-plane proxy for pelvis rotation -- a single 2D camera can't measure true 3D pelvis
  orientation, the same camera-geometry limitation already documented elsewhere in this
  project).

`quality/skill_rules.py` (new) takes phase-mean values of these two series (via the existing
`split_phases` -- Katsumi's "backswing"/"impact" map onto this project's existing
"backswing"/"contact" phases) and applies:

- **Twist reversal**: skilled players' shoulder-pelvis twist stays the same sign from backswing
  (-18.9 +/- 13.1 deg) to contact (-15.5 +/- 17.3 deg); less-skilled players' flips from near-zero
  backswing (-7.7 +/- 26.2 deg) to positive at contact (+10.1 +/- 17.0 deg) -- "separation, which
  had not been established during preparation, may have been produced belatedly" (paper's own
  explanation). Flagged when the query clip's two phases have different signs.
- **Excessive pelvic rotation**: less-skilled players rotate the pelvis further at both
  backswing (skilled -29.6 +/- 12.6 deg vs. less-skilled -46.7 +/- 18.4 deg) and contact
  (skilled -37.1 +/- 24.0 deg vs. less-skilled -65.7 +/- 17.6 deg). Flagged per phase when the
  query clip's value passes the midpoint between the two groups' means for that phase -- a
  directional lean, not a diagnostic cutoff, since the SDs overlap the groups substantially.

Only applies to `backhand_volley` clips -- not wired into `score_clip`/`score_clip_correlated`
(those are the generic 5-joint framework); this is a standalone function returning its own
flags, "separate from the 15 z-score values" as originally scoped. UI wiring is a follow-up.

### Rule 3: Aydin & Aydemir (2026) volley swing-effort rule -- not wired to real data yet

Elite players' volleys show *lower* peak dominant-hand acceleration (48.12 +/- 26.49 m/s^2)
than amateurs' (57.09 +/- 29.86 m/s^2) -- "amateurs overcompensate for technical deficiencies
with excessive, uncontrolled force," while elites use "a controlled, abbreviated swing" (the
"principle of minimum energy"). This needs a peak-acceleration reading off a *volley* swing, but
`hardware/client/`'s recorded takes (backhand/forehand/serve/ballbounce) don't include a volley
yet, and applying this real-hardware-calibrated threshold to `imu_fusion.py`'s synthetic
skeleton-derived signal would be invalid -- that signal isn't in real accelerometer units and
is, by the existing design's own admission, redundant with the skeleton branch. So this rule is
implemented as a standalone pure function taking an already-computed peak acceleration in
m/s^2, not wired into any pipeline -- ready the moment a real volley recording exists. The SDs
here overlap even more than Katsumi's (26-30 against a ~9 m/s^2 mean gap), so this is
implemented as a lean/note, explicitly not a per-swing classifier.

## Components

### `quality/angles.py` (extend)

- `signed_shoulder_pelvis_twist_series(kpts)`, `signed_pelvic_rotation_series(kpts)` -- both
  `(T,)` radians via `arctan2`, T=0 returns `(0,)` (same empty-input contract as the existing
  series functions).

### `quality/skill_rules.py` (new)

- `KATSUMI_BACKHAND_VOLLEY` -- the (mean, std) degrees table above, with the citation as a
  module comment.
- `check_shoulder_pelvis_twist_reversal(twist_backswing_deg, twist_contact_deg)` -> bool.
- `check_excessive_pelvic_rotation(pelvic_backswing_deg, pelvic_contact_deg)` ->
  `{"backswing": bool, "contact": bool}`.
- `evaluate_backhand_volley_skill_rules(kpts)` -- runs `split_phases` +
  `signed_shoulder_pelvis_twist_series`/`signed_pelvic_rotation_series`, converts to degrees,
  applies both rules above, returns `{"flags": [{"rule": str, "phase": str or None, "note":
  str}, ...]}` (empty list if a phase has no frames to evaluate, same as the rest of `quality/`
  surfaces missing data rather than raising).
- `AYDIN_AYDEMIR_VOLLEY` -- the (mean, std) m/s^2 table, with citation.
- `g_to_mps2(g)` -- `g * 9.80665`.
- `check_volley_swing_effort(peak_accel_mps2)` -> `{"flagged": bool, "note": str or None}`.

## Testing

- `quality/test_angles.py`: signed series functions against hand-placed keypoints with a known
  sign and magnitude (both a positive and a negative case, since this is exactly what
  `arccos`-based `trunk_rotation_series` couldn't do).
- `quality/test_skill_rules.py`: each rule function against values on both sides of its
  threshold; `evaluate_backhand_volley_skill_rules` against a synthetic clip with known
  backswing/contact geometry (same construction technique as
  `test_build_expert_templates.py`'s `synthetic_clip`); `g_to_mps2` numeric check.

## Out of scope

- Wiring any of this into `app.py`'s UI or into `score_clip`/`score_clip_correlated`.
- Forehand volley (paper's own scope is BV-only) and any equivalent rule for
  forehand/backhand/serve (candidate paper read, not usable -- see Sourcing note above).
- Collecting a real volley IMU recording (needed before `check_volley_swing_effort` can run on
  anything real).
