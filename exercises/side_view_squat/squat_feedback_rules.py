"""Lenient, explainable feedback rules for completed side-view squat reps."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

try:
    from .config import SQUAT_THRESHOLDS
except ImportError:
    from config import SQUAT_THRESHOLDS


FEEDBACK = {
    "shallow_depth": "Try going slightly lower.",
    "incomplete_lockout": "Stand fully between repetitions.",
    "heel_lift": "Keep your heels grounded.",
    "excessive_torso_lean": "Keep your chest a little more upright.",
    "chest_collapse": "Drive your hips and chest up together.",
    "uncontrolled_tempo": "Slow down and control the descent.",
    "general": "Try to keep the repetition smooth and controlled.",
}


@dataclass(frozen=True)
class RuleResult:
    name: str
    feedback: str
    severity: float
    confidence: float
    evidence: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _clamp(value: float) -> float:
    return max(0.0, min(float(value), 1.0))


def evaluate_rules(
    features: Mapping[str, float],
    frames: Sequence[Mapping] | None = None,
    baseline: Mapping[str, float] | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> list[RuleResult]:
    t = dict(SQUAT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    frames = list(frames or [])
    baseline = baseline or {}
    results: list[RuleResult] = []
    reliability = _clamp(float(features.get("pose_visibility_mean", 0.0)))

    shallow_signals = [
        features["knee_angle_at_bottom"] > t["shallow_knee_angle"],
        features["knee_rom"] < t["shallow_min_rom"],
        features["hip_knee_depth_at_bottom"] < t["shallow_hip_knee_margin"],
        features["hip_vertical_drop"] < t["shallow_min_hip_drop"],
    ]
    shallow_count = sum(shallow_signals)
    if shallow_count >= 2:
        severity = max(
            (features["knee_angle_at_bottom"] - t["shallow_knee_angle"]) / 28.0,
            (t["shallow_min_rom"] - features["knee_rom"]) / 30.0,
            (t["shallow_hip_knee_margin"] - features["hip_knee_depth_at_bottom"]) / 0.20,
        )
        results.append(RuleResult(
            "shallow_depth", FEEDBACK["shallow_depth"], _clamp(severity),
            _clamp((0.64 + 0.09 * shallow_count) * reliability),
            {"signal_count": float(shallow_count), "bottom_knee_angle": features["knee_angle_at_bottom"], "knee_rom": features["knee_rom"]},
        ))

    if frames and baseline:
        final_window = frames[-max(3, int(t["return_confirm_frames"])) :]
        final_knee = sum(float(frame["knee_angle"]) for frame in final_window) / len(final_window)
        final_hip = sum(float(frame["hip_angle"]) for frame in final_window) / len(final_window)
        knee_delta = float(baseline["knee_angle"]) - final_knee
        hip_delta = float(baseline["hip_angle"]) - final_hip
        if knee_delta > t["lockout_knee_delta"] and hip_delta > t["lockout_hip_delta"]:
            results.append(RuleResult(
                "incomplete_lockout", FEEDBACK["incomplete_lockout"],
                _clamp(max(knee_delta / 25.0, hip_delta / 30.0)), _clamp(0.88 * reliability),
                {"final_knee_delta": knee_delta, "final_hip_delta": hip_delta},
            ))

    if (
        features["heel_lift_max"] >= t["heel_lift_threshold"]
        and features["heel_lift_bottom_fraction"] >= t["heel_lift_bottom_fraction"]
        and features["pose_visibility_min"] >= t["reliable_visibility_min"]
    ):
        results.append(RuleResult(
            "heel_lift", FEEDBACK["heel_lift"],
            _clamp((features["heel_lift_max"] - t["heel_lift_threshold"]) / 0.08 + 0.45),
            _clamp((0.72 + 0.25 * features["heel_lift_bottom_fraction"]) * reliability),
            {"heel_lift_max": features["heel_lift_max"], "bottom_fraction": features["heel_lift_bottom_fraction"]},
        ))

    torso_change = features["max_torso_lean"] - features["standing_torso_lean"]
    if features["max_torso_lean"] > t["maximum_torso_lean"] and torso_change > t["maximum_torso_change"]:
        results.append(RuleResult(
            "excessive_torso_lean", FEEDBACK["excessive_torso_lean"],
            _clamp(max((features["max_torso_lean"] - t["maximum_torso_lean"]) / 25.0, (torso_change - t["maximum_torso_change"]) / 25.0)),
            _clamp(0.86 * reliability),
            {"maximum_torso_lean": features["max_torso_lean"], "change_from_standing": torso_change},
        ))

    ascent_frames = [frame for frame in frames if frame.get("phase") == "ASCENDING"]
    ascent_lean_change = 0.0
    if ascent_frames:
        ascent_lean_change = max(float(frame["torso_lean"]) for frame in ascent_frames) - float(ascent_frames[0]["torso_lean"])
    if (
        features["hip_shoulder_rise_difference"] > t["chest_collapse_rise_difference"]
        and ascent_lean_change > t["chest_collapse_lean_change"]
    ):
        results.append(RuleResult(
            "chest_collapse", FEEDBACK["chest_collapse"],
            _clamp(max((features["hip_shoulder_rise_difference"] - t["chest_collapse_rise_difference"]) / 0.15, (ascent_lean_change - t["chest_collapse_lean_change"]) / 20.0)),
            _clamp(0.90 * reliability),
            {"rise_difference": features["hip_shoulder_rise_difference"], "ascent_lean_change": ascent_lean_change},
        ))

    tempo_signals = [
        features["rep_duration"] < t["uncontrolled_rep_duration"],
        features["descent_duration"] < t["uncontrolled_descent_duration"],
        features["max_descent_knee_velocity"] > t["uncontrolled_descent_velocity"],
    ]
    if sum(tempo_signals) >= 2:
        results.append(RuleResult(
            "uncontrolled_tempo", FEEDBACK["uncontrolled_tempo"],
            _clamp(max((t["uncontrolled_rep_duration"] - features["rep_duration"]) / 0.5, (features["max_descent_knee_velocity"] - t["uncontrolled_descent_velocity"]) / 400.0)),
            _clamp((0.72 + 0.08 * sum(tempo_signals)) * reliability),
            {"signal_count": float(sum(tempo_signals)), "rep_duration": features["rep_duration"], "descent_duration": features["descent_duration"]},
        ))

    return sorted(results, key=lambda result: result.severity * result.confidence, reverse=True)


def primary_rule(results: Sequence[RuleResult]) -> RuleResult | None:
    return results[0] if results else None
