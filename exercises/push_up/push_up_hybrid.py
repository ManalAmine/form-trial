"""Combine the quality model with reliable push-up error rules."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

try:
    from .config import FEEDBACK_LABELS, PUSH_UP_THRESHOLDS
    from .push_up_features import ordered_feature_vector
    from .push_up_rules import evaluate_rules, strongest_rules
except ImportError:
    from config import FEEDBACK_LABELS, PUSH_UP_THRESHOLDS
    from push_up_features import ordered_feature_vector
    from push_up_rules import evaluate_rules, strongest_rules


def predict_good_probability(model, features: Mapping[str, float]) -> float:
    values = np.asarray([ordered_feature_vector(features)], dtype=np.float32)
    probabilities = model.predict_proba(values)[0] if hasattr(model, "predict_proba") else None
    if probabilities is None:
        return float(int(model.predict(values)[0]) == 1)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"Quality model does not contain GOOD class 1: {classes}")
    return float(probabilities[classes.index(1)])


def assess_completed_rep(model, features: Mapping[str, float], rep_number=None, thresholds=None) -> dict:
    t = {**PUSH_UP_THRESHOLDS, **(thresholds or {})}
    reliability = features["mean_landmark_visibility"] * features["valid_frame_ratio"]
    if (
        features["mean_landmark_visibility"] < t["reliable_visibility_mean"]
        or features["min_landmark_visibility"] < t["reliable_visibility_min"]
        or features["valid_frame_ratio"] < t["minimum_valid_frame_ratio"]
        or features["frame_count"] < t["minimum_valid_frames"]
    ):
        return {
            "rep_number": rep_number, "quality_probability": None, "ml_quality": "UNSCORED",
            "final_quality": "UNSCORED", "active_errors": [], "strongest_error": None,
            "error_severity": 0.0, "feedback": "Make sure your full body is visible",
            "rep_duration": features["rep_duration"], "pose_reliability": round(reliability, 4),
        }

    probability = predict_good_probability(model, features)
    ml_quality = "GOOD" if probability >= t["ml_good_threshold"] else "BAD"
    rules = evaluate_rules(features, t)
    top_rules = strongest_rules(rules, 2)
    final_good = ml_quality == "GOOD" and not rules
    final_quality = "GOOD" if final_good else "BAD"
    strongest = rules[0] if rules else None
    if final_good:
        feedback = FEEDBACK_LABELS["good"]
    elif strongest:
        feedback = strongest["feedback"]
    else:
        feedback = FEEDBACK_LABELS["general"]
    return {
        "rep_number": rep_number,
        "quality_probability": round(probability, 4),
        "ml_quality": ml_quality,
        "final_quality": final_quality,
        "active_errors": [dict(result) for result in top_rules],
        "strongest_error": strongest["name"] if strongest else None,
        "error_severity": strongest["severity"] if strongest else 0.0,
        "feedback": feedback,
        "rep_duration": features["rep_duration"],
        "pose_reliability": round(reliability, 4),
    }

