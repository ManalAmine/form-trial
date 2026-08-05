"""Geometry and smoothing helpers for side-view push-ups."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def point_xy(point) -> tuple[float, float]:
    if isinstance(point, Mapping):
        return float(point["x"]), float(point["y"])
    if hasattr(point, "x"):
        return float(point.x), float(point.y)
    return float(point[0]), float(point[1])


def visibility(point) -> float:
    if isinstance(point, Mapping):
        return float(point.get("visibility", 1.0))
    return float(getattr(point, "visibility", 1.0))


def euclidean_distance(a, b) -> float:
    ax, ay = point_xy(a)
    bx, by = point_xy(b)
    return float(math.hypot(ax - bx, ay - by))


def calculate_angle(a, b, c) -> float | None:
    """Return the smaller 2-D angle ABC, or None for a zero-length vector."""
    ax, ay = point_xy(a)
    bx, by = point_xy(b)
    cx, cy = point_xy(c)
    first = math.hypot(ax - bx, ay - by)
    second = math.hypot(cx - bx, cy - by)
    if first <= 1e-8 or second <= 1e-8:
        return None
    cosine = ((ax - bx) * (cx - bx) + (ay - by) * (cy - by)) / (first * second)
    return float(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))


def signed_normalized_hip_offset(shoulder, hip, ankle) -> float | None:
    """Return mirror-invariant hip offset; positive is sag and negative is pike.

    Image y increases downward. Orienting the 2-D cross product by the
    shoulder-to-ankle x direction makes the sign invariant when the image or
    facing direction is mirrored.
    """
    sx, sy = point_xy(shoulder)
    hx, hy = point_xy(hip)
    ax, ay = point_xy(ankle)
    dx, dy = ax - sx, ay - sy
    length = math.hypot(dx, dy)
    if length <= 1e-8:
        return None
    cross_distance = (dx * (hy - sy) - dy * (hx - sx)) / length
    orientation = 1.0 if dx >= 0.0 else -1.0
    return float((cross_distance * orientation) / length)


def median_filter(values: Sequence[float], window: int = 5) -> list[float]:
    data = [float(value) for value in values]
    if window <= 1:
        return data
    half = window // 2
    return [
        float(np.median(data[max(0, index - half) : min(len(data), index + half + 1)]))
        for index in range(len(data))
    ]


def finite_velocity(values: Sequence[float], timestamps: Sequence[float]) -> list[float]:
    values = list(map(float, values))
    timestamps = list(map(float, timestamps))
    if not values:
        return []
    result = [0.0]
    for previous, current, start, end in zip(values, values[1:], timestamps, timestamps[1:]):
        dt = end - start
        result.append(0.0 if dt <= 1e-4 else float((current - previous) / dt))
    return result


def robust_mean(values: Sequence[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return 0.0
    if array.size >= 10:
        low, high = np.percentile(array, [10, 90])
        array = array[(array >= low) & (array <= high)]
    return float(np.mean(array))

