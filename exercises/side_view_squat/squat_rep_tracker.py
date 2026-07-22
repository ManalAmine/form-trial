"""Calibrated, hysteretic side-view squat repetition state machine."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np

try:
    from .config import SQUAT_THRESHOLDS
except ImportError:
    from config import SQUAT_THRESHOLDS


class SquatRepTracker:
    PHASES = ("CALIBRATING", "STANDING", "DESCENDING", "BOTTOM", "ASCENDING")

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = dict(SQUAT_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.phase = "CALIBRATING"
        self.baseline: dict[str, float] | None = None
        self.calibration = deque(maxlen=int(self.thresholds["calibration_frames"]))
        self.measurement_history = deque(
            maxlen=int(self.thresholds["measurement_smoothing_window"])
        )
        self.previous_frame = None
        self.transition_count = 0
        self.return_count = 0
        self.rep_frames: list[dict] = []
        self.total_rep_frames = 0
        self.invalid_rep_frames = 0
        self.rep_number = 0
        self.locked_side: str | None = None
        self.minimum_knee = 180.0

    @property
    def active(self) -> bool:
        return self.phase in {"DESCENDING", "BOTTOM", "ASCENDING"}

    def reset_attempt(self, phase: str = "STANDING") -> None:
        self.phase = phase
        self.transition_count = 0
        self.return_count = 0
        self.rep_frames = []
        self.total_rep_frames = 0
        self.invalid_rep_frames = 0
        self.locked_side = None
        self.minimum_knee = 180.0

    def _calibrate(self, frame: dict) -> dict | None:
        if (
            frame["knee_angle"] >= self.thresholds["standing_knee_min"]
            and frame["hip_angle"] >= self.thresholds["standing_hip_min"]
        ):
            self.calibration.append(deepcopy(frame))
        else:
            self.calibration.clear()
        if len(self.calibration) < self.calibration.maxlen:
            return None
        fields = ("knee_angle", "hip_angle", "hip_y", "torso_lean", "body_scale")
        self.baseline = {
            field.replace("_angle", "_angle") if field != "knee_angle" else "knee_angle": float(
                np.median([item[field] for item in self.calibration])
            )
            for field in fields
        }
        self.baseline["hip_angle"] = float(np.median([item["hip_angle"] for item in self.calibration]))
        self.baseline["heel_foot_offset"] = float(
            np.median([item["foot_y"] - item["heel_y"] for item in self.calibration])
        )
        self.phase = "STANDING"
        return {"type": "calibrated", "baseline": deepcopy(self.baseline)}

    def _velocity(self, frame: dict) -> float:
        if not self.previous_frame:
            return 0.0
        dt = frame["timestamp"] - self.previous_frame["timestamp"]
        return 0.0 if dt <= 1e-4 else (frame["knee_angle"] - self.previous_frame["knee_angle"]) / dt

    def _smooth_frame(self, frame: dict) -> dict:
        self.measurement_history.append(deepcopy(frame))
        smoothed = deepcopy(frame)
        fields = (
            "knee_angle", "hip_angle", "ankle_angle", "torso_lean",
            "shoulder_y", "hip_y", "knee_y", "ankle_y", "heel_y", "foot_y",
            "body_scale",
        )
        for field in fields:
            smoothed[field] = float(np.median([item[field] for item in self.measurement_history]))
        return smoothed

    def update(self, frame: dict | None) -> dict:
        event = None
        if frame is None or not frame.get("valid", True):
            if self.active:
                self.total_rep_frames += 1
                self.invalid_rep_frames += 1
                if self.invalid_rep_frames >= self.thresholds["maximum_consecutive_invalid_frames"]:
                    self.reset_attempt()
                    event = {"type": "rejected", "reason": "Pose was lost during the repetition."}
            return {"phase": self.phase, "event": event}

        frame = self._smooth_frame(frame)
        if self.phase == "CALIBRATING":
            event = self._calibrate(frame)
            self.previous_frame = frame
            return {"phase": self.phase, "event": event}

        velocity = self._velocity(frame)
        frame["knee_velocity"] = velocity
        baseline = self.baseline
        assert baseline is not None
        hip_drop = (frame["hip_y"] - baseline["hip_y"]) / max(baseline["body_scale"], 1e-6)
        knee_delta = baseline["knee_angle"] - frame["knee_angle"]

        if self.phase == "STANDING":
            descent_signal = knee_delta >= self.thresholds["descent_knee_delta"] and (
                hip_drop >= self.thresholds["descent_hip_drop"]
                or knee_delta >= self.thresholds["descent_knee_delta"] + 6.0
            )
            self.transition_count = self.transition_count + 1 if descent_signal else 0
            if self.transition_count >= self.thresholds["phase_confirm_frames"]:
                self.phase = "DESCENDING"
                self.locked_side = str(frame.get("side", "left"))
                self.rep_frames = []
                self.total_rep_frames = 0
                self.invalid_rep_frames = 0
                self.minimum_knee = frame["knee_angle"]
                self.transition_count = 0

        if self.active:
            frame["phase"] = self.phase
            self.rep_frames.append(frame)
            self.total_rep_frames += 1
            self.minimum_knee = min(self.minimum_knee, frame["knee_angle"])
            elapsed = self.rep_frames[-1]["timestamp"] - self.rep_frames[0]["timestamp"]
            knee_rom = baseline["knee_angle"] - self.minimum_knee

            if elapsed > self.thresholds["maximum_rep_duration"]:
                self.reset_attempt()
                event = {"type": "rejected", "reason": "The repetition timed out."}
            elif self.phase == "DESCENDING":
                depth_reached = (
                    knee_rom >= self.thresholds["minimum_rep_rom"]
                    and (
                        frame["knee_angle"] <= self.thresholds["bottom_knee_angle"]
                        or hip_drop >= self.thresholds["bottom_hip_drop"]
                    )
                )
                bottom_signal = depth_reached and (
                    abs(velocity) <= self.thresholds["bottom_velocity"]
                    or velocity >= self.thresholds["ascent_velocity"]
                )
                self.transition_count = self.transition_count + 1 if bottom_signal else 0
                if self.transition_count >= self.thresholds["bottom_confirm_frames"]:
                    self.phase = "BOTTOM"
                    for item in self.rep_frames[-self.transition_count :]:
                        item["phase"] = "BOTTOM"
                    self.transition_count = 0
                elif velocity > self.thresholds["ascent_velocity"] and knee_rom < self.thresholds["minimum_rep_rom"]:
                    # A tiny bend returning upward is not a repetition and is silently discarded.
                    self.reset_attempt()
            elif self.phase == "BOTTOM":
                ascent_signal = velocity >= self.thresholds["ascent_velocity"]
                self.transition_count = self.transition_count + 1 if ascent_signal else 0
                if self.transition_count >= self.thresholds["ascent_confirm_frames"]:
                    self.phase = "ASCENDING"
                    for item in self.rep_frames[-self.transition_count :]:
                        item["phase"] = "ASCENDING"
                    self.transition_count = 0
            elif self.phase == "ASCENDING":
                returned = (
                    frame["knee_angle"] >= baseline["knee_angle"] - self.thresholds["return_knee_tolerance"]
                    and frame["hip_angle"] >= baseline["hip_angle"] - self.thresholds["return_hip_tolerance"]
                )
                self.return_count = self.return_count + 1 if returned else 0
                if self.return_count >= self.thresholds["return_confirm_frames"]:
                    duration = self.rep_frames[-1]["timestamp"] - self.rep_frames[0]["timestamp"]
                    valid_frames = sum(item.get("valid", True) for item in self.rep_frames)
                    valid_ratio = valid_frames / max(self.total_rep_frames, 1)
                    if (
                        duration < self.thresholds["minimum_rep_duration"]
                        or valid_frames < self.thresholds["minimum_valid_frames"]
                        or valid_ratio < self.thresholds["minimum_valid_frame_ratio"]
                    ):
                        event = {"type": "rejected", "reason": "The repetition was too short or unreliable."}
                    else:
                        self.rep_number += 1
                        event = {
                            "type": "completed",
                            "rep_number": self.rep_number,
                            "frames": deepcopy(self.rep_frames),
                            "total_frames": self.total_rep_frames,
                            "baseline": deepcopy(baseline),
                            "side": self.locked_side,
                        }
                    self.reset_attempt()

        self.previous_frame = frame
        return {"phase": self.phase, "event": event}
