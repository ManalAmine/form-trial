"""Geometry helpers shared by collection, live inference, and tests."""

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


def calculate_angle(a, b, c) -> float:
    """Return the smaller 2-D angle ABC in degrees."""
    ax, ay = point_xy(a)
    bx, by = point_xy(b)
    cx, cy = point_xy(c)
    radians = math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    angle = abs(math.degrees(radians))
    return float(360.0 - angle if angle > 180.0 else angle)


def torso_lean_from_vertical(shoulder, hip) -> float:
    sx, sy = point_xy(shoulder)
    hx, hy = point_xy(hip)
    return float(math.degrees(math.atan2(abs(sx - hx), max(abs(sy - hy), 1e-8))))


def euclidean_distance(a, b) -> float:
    ax, ay = point_xy(a)
    bx, by = point_xy(b)
    return float(math.hypot(ax - bx, ay - by))


def robust_mean(values: Sequence[float], trim_fraction: float = 0.1) -> float:
    array = np.asarray(list(values), dtype=float)
    if not array.size:
        return 0.0
    array = np.sort(array[np.isfinite(array)])
    if not array.size:
        return 0.0
    trim = int(array.size * trim_fraction)
    if trim and array.size > 2 * trim:
        array = array[trim:-trim]
    return float(np.mean(array))


def median_filter(values: Sequence[float], window: int = 5) -> list[float]:
    values = list(map(float, values))
    if window <= 1 or len(values) < 2:
        return values
    half = window // 2
    return [
        float(np.median(values[max(0, index - half) : min(len(values), index + half + 1)]))
        for index in range(len(values))
    ]


def finite_velocity(values: Sequence[float], timestamps: Sequence[float]) -> list[float]:
    values = median_filter(values)
    velocities = [0.0]
    for previous, current, start, end in zip(values, values[1:], timestamps, timestamps[1:]):
        dt = float(end) - float(start)
        velocities.append(0.0 if dt <= 1e-4 else float((current - previous) / dt))
    return velocities
