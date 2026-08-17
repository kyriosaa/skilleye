"""
Smoke check for Module A's correlated (conditional) z-score scoring
(score_clip_correlated) -- mirrors smoke_check_quality_scoring.py's check for
the original independent scorer, applied to the new one. Same directional
claim, same standard: held-out expert clips should score higher on average
than held-out beginner clips, per stroke class, against that stroke's
covariance-based template. If any stroke fails this, the scoring isn't ready
to demo, independent-vs-correlated.

Usage:
    python smoke_check_correlated_quality_scoring.py --skeletons ../../skeletons \
        --templates ../results/quality_templates/templates.json
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from skeleton_records import load_records
from quality.score import score_clip_correlated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--templates", required=True)
    args = ap.parse_args()

    with open(args.templates) as f:
        data = json.load(f)
    if "covariance" not in data:
        raise SystemExit(
            f"{args.templates} has no 'covariance' key -- rebuild it with "
            "build_expert_templates.py (Module A) before running this check.")
    covariance = data["covariance"]
    val_subjects = set(data["val_subjects"])

    records, _ = load_records(args.skeletons)
    val_records = [r for r in records if r["subject_id"] in val_subjects]
    print(f"scoring {len(val_records)} held-out clips ({len(val_subjects)} subjects) "
          f"with score_clip_correlated()")

    scores_by_stroke_skill = defaultdict(list)
    for r in val_records:
        result = score_clip_correlated(r["kpts"], r["stroke"], covariance)
        scores_by_stroke_skill[(r["stroke"], r["skill_level"])].append(result["overall_score"])

    strokes = sorted({stroke for stroke, _ in scores_by_stroke_skill})
    print(f"\n{'stroke':16s} {'expert mean':>12s} {'beginner mean':>14s} {'experts higher?':>16s}")
    all_ok = True
    for stroke in strokes:
        expert_scores = scores_by_stroke_skill.get((stroke, "expert"), [])
        beginner_scores = scores_by_stroke_skill.get((stroke, "beginner"), [])
        if not expert_scores or not beginner_scores:
            print(f"{stroke:16s} -- insufficient held-out data for one group, skipped --")
            continue
        expert_mean = float(np.mean(expert_scores))
        beginner_mean = float(np.mean(beginner_scores))
        ok = expert_mean > beginner_mean
        all_ok = all_ok and ok
        print(f"{stroke:16s} {expert_mean:12.1f} {beginner_mean:14.1f} {'yes' if ok else 'NO':>16s}")

    print()
    if all_ok:
        print("OVERALL: experts scored higher on every stroke with held-out data "
              "(correlated z-score).")
    else:
        print("OVERALL: at least one stroke did NOT show experts scoring higher with "
              "the correlated z-score -- revisit before using this in the demo.")


if __name__ == "__main__":
    main()
