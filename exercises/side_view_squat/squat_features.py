"""Canonical repetition-level feature extraction for side-view squats."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

try:
    from .config import FEATURE_COLUMNS, SQUAT_THRESHOLDS
    from .squat_geometry import finite_velocity, median_filter, robust_mean
except ImportError:  # Direct script execution.
    from config import FEATURE_COLUMNS, SQUAT_THRESHOLDS
    from squat_geometry import finite_velocity, median_filter, robust_mean


def _values(frames: Sequence[Mapping], name: str, default: float = 0.0) -> list[float]:
    return [float(frame.get(name, default)) for frame in frames]


def _phase_indices(frames: Sequence[Mapping], phase: str) -> list[int]:
    return [index for index, frame in enumerate(frames) if frame.get("phase") == phase]


def _duration(timestamps: Sequence[float], indices: Sequence[int]) -> float:
    return 0.0 if len(indices) < 2 else max(float(timestamps[indices[-1]] - timestamps[indices[0]]), 0.0)


def _movement_smoothness(angles: Sequence[float], timestamps: Sequence[float]) -> float:
    velocities = finite_velocity(angles, timestamps)
    accelerations = finite_velocity(velocities, timestamps)
    if len(accelerations) < 3:
        return 0.0
    # Robust acceleration variability; lower is smoother. Clipping prevents one
    # bad timestamp from dominating a repetition-level feature.
    return float(np.percentile(np.abs(np.asarray(accelerations)), 75))


def build_rep_features(
    frames: Sequence[Mapping],
    baseline: Mapping[str, float],
    total_frames: int | None = None,
) -> dict[str, float]:
    """Build one fixed-order numerical row from a completed repetition."""
    valid_frames = [frame for frame in frames if frame.get("valid", True)]
    if len(valid_frames) < 2:
        raise ValueError("A repetition needs at least two valid frames.")

    timestamps = _values(valid_frames, "timestamp")
    knees = median_filter(_values(valid_frames, "knee_angle"))
    hips = median_filter(_values(valid_frames, "hip_angle"))
    ankles = median_filter(_values(valid_frames, "ankle_angle"))
    torso = median_filter(_values(valid_frames, "torso_lean"))
    body_scale = max(float(baseline.get("body_scale", 0.5)), 1e-6)

    bottom_indices = _phase_indices(valid_frames, "BOTTOM")
    if not bottom_indices:
        minimum_index = int(np.argmin(knees))
        bottom_indices = list(range(max(0, minimum_index - 2), min(len(valid_frames), minimum_index + 3)))
    bottom_index = int(bottom_indices[len(bottom_indices) // 2])

    knee_velocity = finite_velocity(knees, timestamps)
    descent_indices = _phase_indices(valid_frames, "DESCENDING")
    ascent_indices = _phase_indices(valid_frames, "ASCENDING")
    descent_velocities = [knee_velocity[index] for index in descent_indices] or knee_velocity
    ascent_velocities = [knee_velocity[index] for index in ascent_indices] or knee_velocity

    hip_y = median_filter(_values(valid_frames, "hip_y"))
    knee_y = median_filter(_values(valid_frames, "knee_y"))
    shoulder_y = median_filter(_values(valid_frames, "shoulder_y"))
    heel_y = median_filter(_values(valid_frames, "heel_y"))
    foot_y = median_filter(_values(valid_frames, "foot_y"))
    heel_offsets = [foot - heel for foot, heel in zip(foot_y, heel_y)]
    baseline_heel = float(baseline.get("heel_foot_offset", robust_mean(heel_offsets[:3])))
    heel_lifts = [max((offset - baseline_heel) / body_scale, 0.0) for offset in heel_offsets]

    bottom_heel_lifts = [heel_lifts[index] for index in bottom_indices]
    lift_threshold = float(SQUAT_THRESHOLDS["heel_lift_threshold"])
    heel_lift_bottom_fraction = (
        sum(value >= lift_threshold for value in bottom_heel_lifts) / len(bottom_heel_lifts)
        if bottom_heel_lifts
        else 0.0
    )

    hip_shoulder_difference = 0.0
    if ascent_indices:
        start = ascent_indices[0]
        for index in ascent_indices:
            hip_rise = hip_y[start] - hip_y[index]
            shoulder_rise = shoulder_y[start] - shoulder_y[index]
            hip_shoulder_difference = max(
                hip_shoulder_difference,
                (hip_rise - shoulder_rise) / body_scale,
            )

    descent_duration = _duration(timestamps, descent_indices)
    bottom_duration = _duration(timestamps, bottom_indices)
    ascent_duration = _duration(timestamps, ascent_indices)
    rep_duration = max(timestamps[-1] - timestamps[0], 0.0)
    total_frames = int(total_frames if total_frames is not None else len(frames))
    visibility_values = _values(valid_frames, "visibility")

    feature_row = {
        "min_knee_angle": min(knees),
        "max_knee_angle": max(knees),
        "knee_rom": max(knees) - min(knees),
        "knee_angle_at_bottom": robust_mean([knees[index] for index in bottom_indices]),
        "max_descent_knee_velocity": abs(min(descent_velocities)),
        "max_ascent_knee_velocity": max(max(ascent_velocities), 0.0),
        "min_hip_angle": min(hips),
        "max_hip_angle": max(hips),
        "hip_rom": max(hips) - min(hips),
        "hip_angle_at_bottom": robust_mean([hips[index] for index in bottom_indices]),
        "hip_vertical_drop": max((value - float(baseline["hip_y"])) / body_scale for value in hip_y),
        "hip_knee_depth_at_bottom": robust_mean(
            [(hip_y[index] - knee_y[index]) / body_scale for index in bottom_indices]
        ),
        "min_ankle_angle": min(ankles),
        "max_ankle_angle": max(ankles),
        "ankle_rom": max(ankles) - min(ankles),
        "ankle_angle_at_bottom": robust_mean([ankles[index] for index in bottom_indices]),
        "standing_torso_lean": float(baseline["torso_lean"]),
        "torso_lean_at_bottom": robust_mean([torso[index] for index in bottom_indices]),
        "max_torso_lean": max(torso),
        "torso_lean_range": max(torso) - min(torso),
        "heel_lift_max": max(heel_lifts),
        "heel_lift_bottom_fraction": heel_lift_bottom_fraction,
        "hip_shoulder_rise_difference": hip_shoulder_difference,
        "movement_smoothness": _movement_smoothness(knees, timestamps),
        "rep_duration": rep_duration,
        "descent_duration": descent_duration,
        "bottom_duration": bottom_duration,
        "ascent_duration": ascent_duration,
        "descent_ascent_ratio": descent_duration / max(ascent_duration, 1e-6),
        "pose_visibility_mean": float(np.mean(visibility_values)),
        "pose_visibility_min": float(np.min(visibility_values)),
        "valid_frame_ratio": len(valid_frames) / max(total_frames, 1),
        "valid_frame_count": len(valid_frames),
    }
    return {name: round(float(feature_row[name]), 6) for name in FEATURE_COLUMNS}


def ordered_feature_vector(feature_row: Mapping[str, float]) -> list[float]:
    missing = [name for name in FEATURE_COLUMNS if name not in feature_row]
    if missing:
        raise ValueError(f"Missing squat features: {missing}")
    return [float(feature_row[name]) for name in FEATURE_COLUMNS]
