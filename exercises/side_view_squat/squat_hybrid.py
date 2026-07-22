"""Combine the binary quality model with the strongest reliable feedback rule."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

try:
    from .config import FEATURE_COLUMNS, SQUAT_THRESHOLDS
    from .squat_feedback_rules import FEEDBACK, evaluate_rules, primary_rule
    from .squat_features import ordered_feature_vector
except ImportError:
    from config import FEATURE_COLUMNS, SQUAT_THRESHOLDS
    from squat_feedback_rules import FEEDBACK, evaluate_rules, primary_rule
    from squat_features import ordered_feature_vector


def predict_good_probability(model, feature_row: Mapping[str, float]) -> float:
    values = np.asarray([ordered_feature_vector(feature_row)], dtype=np.float32)
    if not hasattr(model, "predict_proba"):
        return float(int(model.predict(values)[0]) == 1)
    probabilities = model.predict_proba(values)[0]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Quality model does not contain class 1: {classes}")
    return float(probabilities[classes.index(1)])


def assess_completed_rep(
    model,
    feature_row: Mapping[str, float],
    frames: Sequence[Mapping] | None = None,
    baseline: Mapping[str, float] | None = None,
    rep_number: int | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict:
    t = dict(SQUAT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    reliability = float(feature_row["pose_visibility_mean"]) * float(feature_row["valid_frame_ratio"])
    if (
        feature_row["pose_visibility_mean"] < t["reliable_visibility_mean"]
        or feature_row["pose_visibility_min"] < t["reliable_visibility_min"]
        or feature_row["valid_frame_ratio"] < t["minimum_valid_frame_ratio"]
        or feature_row["valid_frame_count"] < t["minimum_valid_frames"]
    ):
        return {
            "rep_number": rep_number,
            "quality": "UNSCORED",
            "good_probability": None,
            "primary_error": None,
            "feedback": "Move so your full body is visible.",
            "rule_confidence": 0.0,
            "pose_reliability": round(reliability, 4),
            "rep_duration_seconds": feature_row["rep_duration"],
            "decision_reason": "unreliable_pose",
            "triggered_rules": [],
        }

    probability = predict_good_probability(model, feature_row)
    rules = evaluate_rules(feature_row, frames, baseline, t)
    strongest = primary_rule(rules)
    strong_rule = (
        strongest is not None
        and strongest.confidence >= t["strong_rule_confidence"]
        and strongest.severity >= t["strong_rule_severity"]
    )
    reportable_rule = (
        strongest is not None
        and strongest.confidence >= t["minimum_feedback_rule_confidence"]
        and strongest.severity >= t["minimum_feedback_rule_severity"]
    )

    if probability >= t["good_probability_threshold"]:
        quality, reason = "GOOD", "model_confident_good"
    elif probability <= t["bad_probability_threshold"]:
        quality, reason = "BAD", "model_confident_bad"
    elif strong_rule:
        quality, reason = "BAD", "strong_rule_override"
    else:
        quality, reason = "BORDERLINE", "borderline_model_no_strong_rule"

    if quality == "GOOD":
        feedback, error, confidence = "Good rep.", None, 0.0
    elif quality == "BAD" and reportable_rule:
        feedback, error, confidence = strongest.feedback, strongest.name, strongest.confidence
    elif quality == "BAD":
        feedback, error, confidence = FEEDBACK["general"], "general", 0.0
    else:
        feedback, error, confidence = "Rep counted; no strong correction.", None, 0.0

    return {
        "rep_number": rep_number,
        "quality": quality,
        "good_probability": round(probability, 4),
        "primary_error": error,
        "feedback": feedback,
        "rule_confidence": round(confidence, 4),
        "pose_reliability": round(reliability, 4),
        "rep_duration_seconds": feature_row["rep_duration"],
        "decision_reason": reason,
        "triggered_rules": [result.to_dict() for result in rules],
    }
