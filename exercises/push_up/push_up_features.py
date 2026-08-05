"""Canonical repetition-level feature extraction for side-view push-ups."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

try:
    from .config import FEATURE_COLUMNS, PUSH_UP_THRESHOLDS
    from .push_up_geometry import finite_velocity, median_filter, robust_mean
except ImportError:
    from config import FEATURE_COLUMNS, PUSH_UP_THRESHOLDS
    from push_up_geometry import finite_velocity, median_filter, robust_mean


def _values(frames: Sequence[Mapping], name: str, default: float = 0.0) -> list[float]:
    return [float(frame.get(name, default)) for frame in frames]


def _indices(frames: Sequence[Mapping], phase: str) -> list[int]:
    return [index for index, frame in enumerate(frames) if frame.get("phase") == phase]


def _duration(timestamps: Sequence[float], indices: Sequence[int]) -> float:
    return 0.0 if len(indices) < 2 else max(float(timestamps[indices[-1]] - timestamps[indices[0]]), 0.0)


def _correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) < 3 or np.std(first) <= 1e-8 or np.std(second) <= 1e-8:
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else 0.0


def build_rep_features(
    frames: Sequence[Mapping],
    total_frames: int | None = None,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Build one deterministic, normalized feature row for a completed attempt."""
    t = dict(PUSH_UP_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    valid_frames = [frame for frame in frames if frame.get("valid", True)]
    if len(valid_frames) < 2:
        raise ValueError("A push-up repetition needs at least two valid frames.")

    window = int(t["smoothing_window_size"])
    timestamps = _values(valid_frames, "timestamp")
    elbows = median_filter(_values(valid_frames, "elbow_angle"), window)
    body_angles = median_filter(_values(valid_frames, "body_line_angle"), window)
    hip_offsets = median_filter(_values(valid_frames, "signed_hip_offset"), window)
    shoulder_y = median_filter(_values(valid_frames, "shoulder_y"), window)
    hip_y = median_filter(_values(valid_frames, "hip_y"), window)
    body_scales = median_filter(_values(valid_frames, "body_scale", 0.5), window)
    body_scale = max(robust_mean(body_scales), 1e-8)
    velocities = finite_velocity(elbows, timestamps)
    accelerations = finite_velocity(velocities, timestamps)

    descent = _indices(valid_frames, "DESCENDING")
    bottom = _indices(valid_frames, "BOTTOM")
    ascent = _indices(valid_frames, "ASCENDING")
    top = _indices(valid_frames, "TOP")
    minimum_index = int(np.argmin(elbows))
    if not bottom:
        bottom = list(range(max(0, minimum_index - 1), min(len(valid_frames), minimum_index + 2)))
    if not descent:
        descent = list(range(0, minimum_index + 1))
    if not ascent:
        ascent = list(range(minimum_index, len(valid_frames)))
    if not top:
        top = list(range(min(3, len(valid_frames))))

    descent_velocities = [velocities[index] for index in descent]
    ascent_velocities = [velocities[index] for index in ascent]
    descent_duration = _duration(timestamps, descent)
    ascent_duration = _duration(timestamps, ascent)
    rep_duration = max(timestamps[-1] - timestamps[0], 0.0)
    bottom_duration = _duration(timestamps, bottom)

    reversal_change = 0.0
    if 0 < minimum_index < len(velocities) - 1:
        before = min(velocities[max(0, minimum_index - 2) : minimum_index + 1])
        after = max(velocities[minimum_index : min(len(velocities), minimum_index + 3)])
        reversal_change = max(after - before, 0.0)

    sag_fraction = sum(value >= t["hip_sag_threshold"] for value in hip_offsets) / len(hip_offsets)
    high_fraction = sum(value <= -t["hips_too_high_threshold"] for value in hip_offsets) / len(hip_offsets)
    shoulder_travel = (max(shoulder_y) - min(shoulder_y)) / body_scale
    hip_travel = (max(hip_y) - min(hip_y)) / body_scale
    visibility_values = _values(valid_frames, "visibility")
    total_frames = int(total_frames if total_frames is not None else len(frames))

    feature_row = {
        "min_elbow_angle": min(elbows),
        "max_elbow_angle": max(elbows),
        "start_elbow_angle": max(elbows[: min(3, len(elbows))]),
        "end_elbow_angle": max(elbows[-min(3, len(elbows)) :]),
        "elbow_rom": max(elbows) - min(elbows),
        "mean_elbow_angle": float(np.mean(elbows)),
        "std_elbow_angle": float(np.std(elbows)),
        "mean_elbow_velocity": float(np.mean(np.abs(velocities))),
        "max_elbow_velocity": max(np.abs(velocities)),
        "max_descent_velocity": abs(min(descent_velocities)),
        "max_ascent_velocity": max(max(ascent_velocities), 0.0),
        "rep_duration": rep_duration,
        "descent_duration": descent_duration,
        "ascent_duration": ascent_duration,
        "bottom_transition_duration": bottom_duration,
        "descent_ascent_ratio": descent_duration / max(ascent_duration, 1e-6),
        "frame_count": len(valid_frames),
        "mean_body_line_angle": float(np.mean(body_angles)),
        "min_body_line_angle": min(body_angles),
        "max_body_line_angle": max(body_angles),
        "mean_signed_hip_offset": float(np.mean(hip_offsets)),
        "min_signed_hip_offset": min(hip_offsets),
        "max_signed_hip_offset": max(hip_offsets),
        "std_hip_offset": float(np.std(hip_offsets)),
        "hip_offset_at_top": robust_mean([hip_offsets[index] for index in top]),
        "hip_offset_at_bottom": robust_mean([hip_offsets[index] for index in bottom]),
        "hip_sag_frame_fraction": sag_fraction,
        "hips_high_frame_fraction": high_fraction,
        "shoulder_vertical_travel": shoulder_travel,
        "hip_vertical_travel": hip_travel,
        "shoulder_hip_travel_difference": abs(shoulder_travel - hip_travel),
        "hip_to_shoulder_travel_ratio": hip_travel / max(shoulder_travel, 1e-6),
        "shoulder_hip_movement_correlation": _correlation(shoulder_y, hip_y),
        "movement_smoothness": float(np.percentile(np.abs(accelerations), 75)) if accelerations else 0.0,
        "bottom_reversal_velocity_change": reversal_change,
        "mean_landmark_visibility": float(np.mean(visibility_values)),
        "min_landmark_visibility": min(visibility_values),
        "valid_frame_ratio": len(valid_frames) / max(total_frames, 1),
    }
    return {name: round(float(feature_row[name]), 6) for name in FEATURE_COLUMNS}


def ordered_feature_vector(feature_row: Mapping[str, float]) -> list[float]:
    missing = [name for name in FEATURE_COLUMNS if name not in feature_row]
    if missing:
        raise ValueError(f"Missing push-up features: {missing}")
    values = [float(feature_row[name]) for name in FEATURE_COLUMNS]
    if not np.isfinite(values).all():
        raise ValueError("Push-up feature vector contains NaN or infinite values.")
    return values
