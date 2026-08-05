"""MediaPipe adaptation, stable side selection, and push-up camera guidance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .config import PUSH_UP_THRESHOLDS
    from .push_up_geometry import (
        calculate_angle,
        euclidean_distance,
        signed_normalized_hip_offset,
        visibility,
    )
except ImportError:
    from config import PUSH_UP_THRESHOLDS
    from push_up_geometry import calculate_angle, euclidean_distance, signed_normalized_hip_offset, visibility


LANDMARK_INDEX = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}
SIDE_PARTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


@dataclass(frozen=True)
class CameraAssessment:
    reliable: bool
    guidance: str
    side: str | None
    facing_direction: str | None
    measurement: dict[str, float] | None
    side_view_ratio: float = 0.0


def landmarks_to_named(landmarks) -> dict[str, Any]:
    if landmarks is None or len(landmarks) <= max(LANDMARK_INDEX.values()):
        raise ValueError("MediaPipe pose landmarks are missing required push-up points.")
    return {name: landmarks[index] for name, index in LANDMARK_INDEX.items()}


def side_visibility(named: dict[str, Any], side: str) -> float:
    try:
        return float(np.mean([visibility(named[f"{side}_{part}"]) for part in SIDE_PARTS]))
    except KeyError:
        return 0.0


def choose_visible_side(
    named: dict[str, Any],
    preferred_side: str | None = None,
    switch_margin: float = 0.12,
    lock_preferred: bool = False,
) -> str:
    if lock_preferred and preferred_side in {"left", "right"}:
        return preferred_side
    scores = {side: side_visibility(named, side) for side in ("left", "right")}
    best = max(scores, key=scores.get)
    if preferred_side in scores and scores[preferred_side] + switch_margin >= scores[best]:
        return preferred_side
    return best


def infer_facing_direction(named: dict[str, Any], side: str) -> str:
    shoulder_x = float(named[f"{side}_shoulder"].x)
    wrist_x = float(named[f"{side}_wrist"].x)
    return "right" if wrist_x >= shoulder_x else "left"


def build_measurement(named: dict[str, Any], side: str, timestamp: float) -> dict[str, float]:
    points = {part: named[f"{side}_{part}"] for part in SIDE_PARTS}
    elbow_angle = calculate_angle(points["shoulder"], points["elbow"], points["wrist"])
    body_line_angle = calculate_angle(points["shoulder"], points["hip"], points["ankle"])
    hip_offset = signed_normalized_hip_offset(points["shoulder"], points["hip"], points["ankle"])
    body_scale = euclidean_distance(points["shoulder"], points["ankle"])
    if elbow_angle is None or body_line_angle is None or hip_offset is None or body_scale <= 1e-8:
        raise ValueError("Degenerate landmark geometry.")
    visibilities = [visibility(points[part]) for part in SIDE_PARTS]
    measurement = {
        "timestamp": float(timestamp),
        "elbow_angle": float(elbow_angle),
        "body_line_angle": float(body_line_angle),
        "signed_hip_offset": float(hip_offset),
        "body_scale": float(body_scale),
        "visibility": float(np.mean(visibilities)),
        "visibility_min": float(min(visibilities)),
        "side": side,
        "facing_direction": infer_facing_direction(named, side),
        "valid": True,
    }
    for part, point in points.items():
        measurement[f"{part}_x"] = float(point.x)
        measurement[f"{part}_y"] = float(point.y)
    return measurement


def assess_camera(
    landmarks,
    timestamp: float,
    preferred_side: str | None = None,
    thresholds: dict[str, float] | None = None,
    lock_side: bool = False,
) -> CameraAssessment:
    t = dict(PUSH_UP_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    if landmarks is None:
        return CameraAssessment(False, "Camera position lost", None, None, None)
    try:
        named = landmarks_to_named(landmarks)
        side = choose_visible_side(
            named,
            preferred_side,
            t["side_switch_visibility_margin"],
            lock_preferred=lock_side,
        )
        measurement = build_measurement(named, side, timestamp)
    except (KeyError, TypeError, ValueError, IndexError):
        return CameraAssessment(False, "Make sure your full body is visible", None, None, None)

    required = [named[f"{side}_{part}"] for part in SIDE_PARTS]
    visibilities = [visibility(point) for point in required]
    if min(visibilities) < t["minimum_core_visibility"] or np.mean(visibilities) < t["minimum_landmark_visibility"]:
        measurement["valid"] = False
        weakest = SIDE_PARTS[int(np.argmin(visibilities))]
        if weakest == "wrist":
            guidance = "Keep your hands visible"
        elif weakest == "ankle":
            guidance = "Keep your feet visible"
        else:
            guidance = "Improve the lighting and keep your full body visible"
        return CameraAssessment(False, guidance, side, measurement["facing_direction"], measurement)

    xs = [float(point.x) for point in required]
    ys = [float(point.y) for point in required]
    margin = t["frame_edge_margin"]
    if min(xs) < margin or max(xs) > 1.0 - margin or min(ys) < margin or max(ys) > 1.0 - margin:
        measurement["valid"] = False
        return CameraAssessment(False, "Make sure your full body is visible", side, measurement["facing_direction"], measurement)

    span = max(max(xs) - min(xs), max(ys) - min(ys))
    if span < t["minimum_body_frame_fraction"]:
        measurement["valid"] = False
        return CameraAssessment(False, "Move closer while keeping your full body visible", side, measurement["facing_direction"], measurement)
    if span > t["maximum_body_frame_fraction"]:
        measurement["valid"] = False
        return CameraAssessment(False, "Move farther from the camera", side, measurement["facing_direction"], measurement)

    shoulder_width = abs(float(named["left_shoulder"].x) - float(named["right_shoulder"].x))
    hip_width = abs(float(named["left_hip"].x) - float(named["right_hip"].x))
    side_ratio = max(shoulder_width, hip_width) / max(measurement["body_scale"], 1e-8)
    if side_ratio > t["maximum_side_width_ratio"]:
        measurement["valid"] = False
        return CameraAssessment(False, "Turn fully sideways", side, measurement["facing_direction"], measurement, side_ratio)

    horizontal_ratio = abs(measurement["ankle_x"] - measurement["shoulder_x"]) / max(measurement["body_scale"], 1e-8)
    measurement["start_position_ok"] = horizontal_ratio >= t["minimum_horizontal_body_ratio"]
    guidance = "Camera position is good"
    if not measurement["start_position_ok"]:
        guidance = "Get into the starting push-up position"
    return CameraAssessment(True, guidance, side, measurement["facing_direction"], measurement, side_ratio)
