"""Explainable repetition-level rules for visible side-view push-up errors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

try:
    from .config import FEEDBACK_LABELS, PUSH_UP_THRESHOLDS
except ImportError:
    from config import FEEDBACK_LABELS, PUSH_UP_THRESHOLDS


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def _result(name, active, severity, confidence, measured, threshold, evidence=None):
    confidence = _clamp(confidence)
    active = bool(active and confidence >= 0.0)
    return {
        "name": name,
        "active": active,
        "severity": round(_clamp(severity), 4),
        "confidence": round(confidence, 4),
        "measured_value": round(float(measured), 6),
        "threshold": float(threshold),
        "feedback": FEEDBACK_LABELS[name],
        "evidence": evidence or {},
    }


def detect_shallow_depth(features: Mapping[str, float], thresholds=None) -> dict:
    t = {**PUSH_UP_THRESHOLDS, **(thresholds or {})}
    angle = float(features["min_elbow_angle"])
    travel = float(features["shoulder_vertical_travel"])
    angle_signal = angle > t["shallow_depth_angle"]
    supporting = travel < t["shallow_shoulder_travel"] or features["elbow_rom"] < t["minimum_full_rom"]
    severity = max((angle - t["shallow_depth_angle"]) / 35.0, 0.0)
    confidence = 0.78 + (0.12 if supporting else 0.0)
    active = angle_signal and confidence >= t["minimum_rule_confidence"]
    return _result("shallow_depth", active, severity, confidence, angle, t["shallow_depth_angle"], {
        "shoulder_vertical_travel": travel, "elbow_rom": float(features["elbow_rom"])
    })


def detect_incomplete_extension(features: Mapping[str, float], thresholds=None) -> dict:
    t = {**PUSH_UP_THRESHOLDS, **(thresholds or {})}
    angle = float(features["end_elbow_angle"])
    gap = t["incomplete_extension_angle"] - angle
    confidence = 0.88 if gap >= 3.0 else 0.0
    return _result(
        "incomplete_extension", gap >= 3.0 and confidence >= t["minimum_rule_confidence"],
        gap / 35.0, confidence, angle, t["incomplete_extension_angle"],
        {"max_elbow_angle": float(features["max_elbow_angle"])},
    )


def detect_hip_sag(features: Mapping[str, float], thresholds=None) -> dict:
    t = {**PUSH_UP_THRESHOLDS, **(thresholds or {})}
    offset = float(features["max_signed_hip_offset"])
    persistence = float(features["hip_sag_frame_fraction"])
    active = offset >= t["hip_sag_threshold"] and persistence >= t["hip_offset_persistence_fraction"]
    confidence = 0.72 + 0.25 * min(persistence, 1.0) if active else 0.0
    return _result("hip_sag", active and confidence >= t["minimum_rule_confidence"],
                   (offset - t["hip_sag_threshold"]) / 0.10, confidence, offset,
                   t["hip_sag_threshold"], {"frame_fraction": persistence})


def detect_hips_too_high(features: Mapping[str, float], thresholds=None) -> dict:
    t = {**PUSH_UP_THRESHOLDS, **(thresholds or {})}
    magnitude = max(-float(features["min_signed_hip_offset"]), 0.0)
    persistence = float(features["hips_high_frame_fraction"])
    active = magnitude >= t["hips_too_high_threshold"] and persistence >= t["hip_offset_persistence_fraction"]
    confidence = 0.72 + 0.25 * min(persistence, 1.0) if active else 0.0
    return _result("hips_too_high", active and confidence >= t["minimum_rule_confidence"],
                   (magnitude - t["hips_too_high_threshold"]) / 0.10, confidence, magnitude,
                   t["hips_too_high_threshold"], {"frame_fraction": persistence})


def detect_uncontrolled_tempo(features: Mapping[str, float], thresholds=None) -> dict:
    t = {**PUSH_UP_THRESHOLDS, **(thresholds or {})}
    signals = [
        features["rep_duration"] < t["uncontrolled_rep_duration"],
        features["descent_duration"] < t["uncontrolled_descent_duration"],
        features["max_elbow_velocity"] > t["uncontrolled_angular_velocity"],
        features["bottom_reversal_velocity_change"] > t["uncontrolled_reversal_change"],
        features["movement_smoothness"] > t["uncontrolled_smoothness"],
    ]
    count = sum(bool(value) for value in signals)
    active = count >= 2
    severity = max(
        (t["uncontrolled_rep_duration"] - features["rep_duration"]) / 0.50,
        (features["max_elbow_velocity"] - t["uncontrolled_angular_velocity"]) / 500.0,
        (features["bottom_reversal_velocity_change"] - t["uncontrolled_reversal_change"]) / 600.0,
        0.0,
    )
    confidence = 0.64 + 0.10 * count if active else 0.0
    return _result("uncontrolled_tempo", active and confidence >= t["minimum_rule_confidence"],
                   severity, confidence, features["rep_duration"], t["uncontrolled_rep_duration"],
                   {"signal_count": float(count)})


def evaluate_rules(features: Mapping[str, float], thresholds=None) -> list[dict]:
    results = [
        detect_shallow_depth(features, thresholds),
        detect_incomplete_extension(features, thresholds),
        detect_hip_sag(features, thresholds),
        detect_hips_too_high(features, thresholds),
        detect_uncontrolled_tempo(features, thresholds),
    ]
    active = [result for result in results if result["active"]]
    sag = next((item for item in active if item["name"] == "hip_sag"), None)
    high = next((item for item in active if item["name"] == "hips_too_high"), None)
    if sag and high:
        weaker = high if sag["severity"] * sag["confidence"] >= high["severity"] * high["confidence"] else sag
        active.remove(weaker)
    return sorted(active, key=lambda item: item["severity"] * item["confidence"], reverse=True)


def strongest_rules(results: Sequence[Mapping], limit: int = 2) -> list[Mapping]:
    return list(results[: max(0, limit)])
