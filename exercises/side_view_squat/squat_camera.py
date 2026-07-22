"""MediaPipe landmark adaptation, side selection, and camera validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .config import SQUAT_THRESHOLDS
    from .squat_geometry import calculate_angle, euclidean_distance, torso_lean_from_vertical, visibility
except ImportError:
    from config import SQUAT_THRESHOLDS
    from squat_geometry import calculate_angle, euclidean_distance, torso_lean_from_vertical, visibility


LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot": 31,
    "right_foot": 32,
}
SIDE_PARTS = ("shoulder", "hip", "knee", "ankle", "heel", "foot")


@dataclass(frozen=True)
class CameraAssessment:
    reliable: bool
    guidance: str
    side: str | None
    measurement: dict[str, float] | None
    side_view_ratio: float = 0.0


def landmarks_to_named(landmarks) -> dict[str, Any]:
    return {name: landmarks[index] for name, index in LANDMARK_INDEX.items()}


def side_visibility(named: dict[str, Any], side: str) -> float:
    return float(np.mean([visibility(named[f"{side}_{part}"]) for part in SIDE_PARTS]))


def choose_visible_side(
    named: dict[str, Any],
    preferred_side: str | None = None,
    switch_margin: float = 0.12,
) -> str:
    scores = {side: side_visibility(named, side) for side in ("left", "right")}
    best = max(scores, key=scores.get)
    if preferred_side in scores and scores[preferred_side] + switch_margin >= scores[best]:
        return preferred_side
    return best


def build_measurement(named: dict[str, Any], side: str, timestamp: float) -> dict[str, float]:
    shoulder = named[f"{side}_shoulder"]
    hip = named[f"{side}_hip"]
    knee = named[f"{side}_knee"]
    ankle = named[f"{side}_ankle"]
    heel = named[f"{side}_heel"]
    foot = named[f"{side}_foot"]
    side_points = [shoulder, hip, knee, ankle, heel, foot]
    body_scale = max(euclidean_distance(shoulder, ankle), 1e-6)

    return {
        "timestamp": float(timestamp),
        "knee_angle": calculate_angle(hip, knee, ankle),
        "hip_angle": calculate_angle(shoulder, hip, knee),
        "ankle_angle": calculate_angle(knee, ankle, foot),
        "torso_lean": torso_lean_from_vertical(shoulder, hip),
        "shoulder_x": float(shoulder.x),
        "shoulder_y": float(shoulder.y),
        "hip_x": float(hip.x),
        "hip_y": float(hip.y),
        "knee_x": float(knee.x),
        "knee_y": float(knee.y),
        "ankle_x": float(ankle.x),
        "ankle_y": float(ankle.y),
        "heel_x": float(heel.x),
        "heel_y": float(heel.y),
        "foot_x": float(foot.x),
        "foot_y": float(foot.y),
        "body_scale": body_scale,
        "visibility": float(np.mean([visibility(point) for point in side_points])),
        "visibility_min": float(min(visibility(point) for point in side_points)),
        "valid": True,
        "side": side,
    }


def assess_camera(
    landmarks,
    timestamp: float,
    preferred_side: str | None = None,
    thresholds: dict[str, float] | None = None,
) -> CameraAssessment:
    thresholds = thresholds or SQUAT_THRESHOLDS
    if landmarks is None:
        return CameraAssessment(False, "Move so your full body is visible.", None, None)

    named = landmarks_to_named(landmarks)
    # Once a rep starts, preferred_side is locked by the tracker. If it becomes
    # occluded, reject the frame instead of silently changing body sides.
    side = preferred_side if preferred_side in {"left", "right"} else choose_visible_side(
        named,
        switch_margin=thresholds["side_switch_visibility_margin"],
    )
    measurement = build_measurement(named, side, timestamp)
    required = [named[f"{side}_{part}"] for part in SIDE_PARTS]
    minimum_visibility = min(visibility(point) for point in required)
    if minimum_visibility < thresholds["visibility_threshold"]:
        measurement["valid"] = False
        return CameraAssessment(False, "Improve the lighting and keep your full side visible.", side, measurement)

    all_points = [named[name] for name in LANDMARK_INDEX]
    xs = [float(point.x) for point in all_points]
    ys = [float(point.y) for point in all_points]
    margin = thresholds["frame_edge_margin"]
    if min(xs) < margin or max(xs) > 1.0 - margin or min(ys) < margin or max(ys) > 1.0 - margin:
        measurement["valid"] = False
        return CameraAssessment(False, "Make sure your feet and shoulders are inside the frame.", side, measurement)

    body_fraction = max(ys) - min(ys)
    if body_fraction < thresholds["minimum_body_frame_fraction"]:
        measurement["valid"] = False
        return CameraAssessment(False, "Move closer while keeping your full body visible.", side, measurement)
    if body_fraction > thresholds["maximum_body_frame_fraction"]:
        measurement["valid"] = False
        return CameraAssessment(False, "Move back so your full body is visible.", side, measurement)

    shoulder_width = abs(float(named["left_shoulder"].x) - float(named["right_shoulder"].x))
    hip_width = abs(float(named["left_hip"].x) - float(named["right_hip"].x))
    side_view_ratio = max(shoulder_width, hip_width) / max(measurement["body_scale"], 1e-6)
    if side_view_ratio > thresholds["maximum_side_width_ratio"]:
        measurement["valid"] = False
        return CameraAssessment(False, "Turn sideways to the camera.", side, measurement, side_view_ratio)

    return CameraAssessment(True, "Camera position is good.", side, measurement, side_view_ratio)
