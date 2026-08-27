# Wire Module A/B into the Demo UI — Design

Date: 2026-08-17
Status: Approved for implementation

## Problem

Module A (correlated z-score) and Module B (skill-level rules) are implemented, tested, and
validated against real data (Sections 2.8-2.9, 3.4 of the README), but only reachable through
their own scripts/functions -- not through `app.py`, the Streamlit demo a judge would actually
click through.

## Approach

Minimal, additive changes to `app.py`, keeping its existing structure:

1. **Fix the hardcoded `E:/SkillEye/...` paths** to be relative to the script's own location
   (`Path(__file__).parent`), so the demo actually runs on whatever machine/drive the repo is
   cloned to -- a pre-existing portability bug, unrelated to Module A/B, fixed while in this file.
2. **Scoring-mode toggle** (sidebar radio: "Independent (v1)" / "Correlated (Module A)") --
   switches between `score_clip()` and `score_clip_correlated()`. Both already share the same
   output shape, so the rest of the display code (table, suggestions) is unchanged either way.
3. **Module B panel**, shown only when the predicted stroke is `backhand_volley` (Module B's
   actual scope, per Section 2.9) -- calls `evaluate_backhand_volley_skill_rules(kpts)` and
   lists any flags with their notes in a new "Skill-level checks" section below the existing
   correction suggestions.

## Out of scope

- The Aydin & Aydemir volley-effort rule (`check_volley_swing_effort`) -- still not wired
  anywhere, since it needs a real ball-contact accelerometer reading this demo (skeleton-only)
  has no access to.
- New tests -- `app.py` has no existing test file (Streamlit UI code, same convention as
  `live_dashboard.py`); this stays consistent with that.
