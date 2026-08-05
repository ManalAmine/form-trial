"""Hysteretic side-view push-up repetition state machine."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

import numpy as np

try:
    from .config import PUSH_UP_THRESHOLDS
except ImportError:
    from config import PUSH_UP_THRESHOLDS


class PushUpStateMachine:
    STATES = ("WAITING_FOR_POSITION", "TOP", "DESCENDING", "BOTTOM", "ASCENDING")

    def __init__(self, thresholds: dict[str, float] | None = None):
        self.thresholds = dict(PUSH_UP_THRESHOLDS)
        if thresholds:
            self.thresholds.update(thresholds)
        self.state = "WAITING_FOR_POSITION"
        self.measurement_history = deque(maxlen=int(self.thresholds["smoothing_window_size"]))
        self.top_preroll = deque(maxlen=max(3, int(self.thresholds["stable_frames"])))
        self.previous_frame: dict | None = None
        self.transition_count = 0
        self.top_count = 0
        self.rep_frames: list[dict] = []
        self.total_rep_frames = 0
        self.consecutive_invalid_frames = 0
        self.rep_number = 0
        self.locked_side: str | None = None
        self.facing_direction: str | None = None
        self.minimum_elbow = 180.0
        self.maximum_return_elbow = 0.0
        self.reached_bottom = False
        self.rep_start_time: float | None = None
        self.descent_start_time: float | None = None
        self.bottom_time: float | None = None
        self.ascent_start_time: float | None = None

    @property
    def phase(self) -> str:
        return self.state

    @property
    def active(self) -> bool:
        return self.state in {"DESCENDING", "BOTTOM", "ASCENDING"}

    def reset_attempt(self, state: str = "WAITING_FOR_POSITION") -> None:
        self.state = state
        self.transition_count = 0
        self.top_count = 0
        self.rep_frames = []
        self.total_rep_frames = 0
        self.consecutive_invalid_frames = 0
        self.locked_side = None
        self.facing_direction = None
        self.minimum_elbow = 180.0
        self.maximum_return_elbow = 0.0
        self.reached_bottom = False
        self.rep_start_time = None
        self.descent_start_time = None
        self.bottom_time = None
        self.ascent_start_time = None

    def _smooth_frame(self, frame: dict) -> dict:
        self.measurement_history.append(deepcopy(frame))
        smoothed = deepcopy(frame)
        fields = (
            "elbow_angle", "body_line_angle", "signed_hip_offset", "body_scale",
            "shoulder_x", "shoulder_y", "hip_x", "hip_y", "ankle_x", "ankle_y",
            "visibility", "visibility_min",
        )
        for field in fields:
            smoothed[field] = float(np.median([item[field] for item in self.measurement_history]))
        return smoothed

    def _velocity(self, frame: dict) -> float:
        if self.previous_frame is None:
            return 0.0
        dt = frame["timestamp"] - self.previous_frame["timestamp"]
        return 0.0 if dt <= 1e-4 else float((frame["elbow_angle"] - self.previous_frame["elbow_angle"]) / dt)

    def _start_attempt(self, frame: dict) -> None:
        self.state = "DESCENDING"
        self.locked_side = str(frame["side"])
        self.facing_direction = str(frame.get("facing_direction", "unknown"))
        self.rep_frames = [deepcopy(item) for item in self.top_preroll]
        for item in self.rep_frames:
            item["phase"] = "TOP"
        self.total_rep_frames = len(self.rep_frames)
        self.rep_start_time = self.rep_frames[0]["timestamp"] if self.rep_frames else frame["timestamp"]
        self.descent_start_time = frame["timestamp"]
        self.minimum_elbow = frame["elbow_angle"]
        self.transition_count = 0

    def _finish(self, frame: dict, completion_reason: str) -> dict:
        rep_end_time = float(frame["timestamp"])
        valid_frames = sum(item.get("valid", True) for item in self.rep_frames)
        valid_ratio = valid_frames / max(self.total_rep_frames, 1)
        duration = rep_end_time - float(rep_end_time if self.rep_start_time is None else self.rep_start_time)
        timing = {
            "rep_start_time": self.rep_start_time,
            "descent_start_time": self.descent_start_time,
            "bottom_time": self.bottom_time,
            "ascent_start_time": self.ascent_start_time,
            "rep_end_time": rep_end_time,
            "descent_duration": None if self.descent_start_time is None or self.ascent_start_time is None else self.ascent_start_time - self.descent_start_time,
            "ascent_duration": None if self.ascent_start_time is None else rep_end_time - self.ascent_start_time,
            "total_duration": duration,
        }
        if (
            duration < self.thresholds["minimum_safe_rep_duration"]
            or valid_frames < self.thresholds["minimum_valid_frames"]
            or valid_ratio < self.thresholds["minimum_valid_frame_ratio"]
        ):
            event = {"type": "rejected", "reason": "The attempt was too short or tracking was unreliable."}
        else:
            self.rep_number += 1
            event = {
                "type": "completed",
                "rep_number": self.rep_number,
                "frames": deepcopy(self.rep_frames),
                "total_frames": self.total_rep_frames,
                "side": self.locked_side,
                "facing_direction": self.facing_direction,
                "reached_bottom": self.reached_bottom,
                "completion_reason": completion_reason,
                "timing": timing,
            }
        self.top_preroll.clear()
        self.reset_attempt("TOP" if completion_reason == "full_extension" else "WAITING_FOR_POSITION")
        return event

    def update(self, frame: dict | None) -> dict:
        event = None
        if frame is None or not frame.get("valid", True):
            if self.active:
                self.total_rep_frames += 1
                self.consecutive_invalid_frames += 1
                if self.consecutive_invalid_frames >= self.thresholds["pose_loss_timeout"]:
                    self.reset_attempt()
                    event = {"type": "rejected", "reason": "Pose was lost during the repetition."}
            return {"state": self.state, "phase": self.state, "event": event}

        self.consecutive_invalid_frames = 0
        frame = self._smooth_frame(frame)
        velocity = self._velocity(frame)
        frame["elbow_velocity"] = velocity
        angle = frame["elbow_angle"]
        top_signal = angle >= self.thresholds["top_elbow_angle"] and bool(frame.get("start_position_ok", True))

        if self.state == "WAITING_FOR_POSITION":
            self.top_count = self.top_count + 1 if top_signal else 0
            if top_signal:
                self.top_preroll.append(deepcopy(frame))
            else:
                self.top_preroll.clear()
            if self.top_count >= self.thresholds["stable_frames"]:
                self.state = "TOP"
                self.top_count = 0

        elif self.state == "TOP":
            if top_signal:
                self.top_preroll.append(deepcopy(frame))
            descent_signal = (
                angle <= min(
                    self.thresholds["descent_start_angle"],
                    self.thresholds["top_elbow_angle"] - self.thresholds["state_hysteresis"],
                )
                and velocity <= -self.thresholds["descent_velocity_threshold"]
            )
            self.transition_count = self.transition_count + 1 if descent_signal else 0
            if self.transition_count >= self.thresholds["stable_frames"]:
                self._start_attempt(frame)

        if self.active:
            frame["phase"] = self.state
            self.rep_frames.append(deepcopy(frame))
            self.total_rep_frames += 1
            self.minimum_elbow = min(self.minimum_elbow, angle)
            elapsed = frame["timestamp"] - float(self.rep_start_time or frame["timestamp"])
            rom = self.thresholds["top_elbow_angle"] - self.minimum_elbow

            if elapsed > self.thresholds["maximum_rep_duration"]:
                self.reset_attempt()
                event = {"type": "rejected", "reason": "The repetition timed out."}
            elif self.state == "DESCENDING":
                full_depth = (
                    angle <= self.thresholds["bottom_elbow_angle"]
                    and rom >= self.thresholds["minimum_full_rom"]
                )
                reversal = velocity >= self.thresholds["ascent_velocity_threshold"]
                bottom_signal = full_depth and (abs(velocity) <= self.thresholds["descent_velocity_threshold"] or reversal)
                if bottom_signal:
                    self.transition_count += 1
                elif reversal and rom >= self.thresholds["minimum_candidate_rom"]:
                    self.transition_count += 1
                else:
                    self.transition_count = 0
                if bottom_signal and self.transition_count >= self.thresholds["bottom_stable_frames"]:
                    self.state = "BOTTOM"
                    self.reached_bottom = True
                    self.bottom_time = frame["timestamp"]
                    for item in self.rep_frames[-self.transition_count:]:
                        item["phase"] = "BOTTOM"
                    self.transition_count = 0
                elif reversal and rom >= self.thresholds["minimum_candidate_rom"]:
                    # A meaningful but shallow down-and-up cycle remains labelable;
                    # small top-to-middle noise is discarded.
                    if self.transition_count >= self.thresholds["ascent_stable_frames"]:
                        self.state = "ASCENDING"
                        self.ascent_start_time = frame["timestamp"]
                        for item in self.rep_frames[-self.transition_count:]:
                            item["phase"] = "ASCENDING"
                        self.transition_count = 0
                elif reversal and rom < self.thresholds["minimum_candidate_rom"]:
                    self.reset_attempt("TOP")

            elif self.state == "BOTTOM":
                ascent_signal = (
                    angle >= self.thresholds["bottom_elbow_angle"] + self.thresholds["state_hysteresis"]
                    and velocity >= self.thresholds["ascent_velocity_threshold"]
                )
                self.transition_count = self.transition_count + 1 if ascent_signal else 0
                if self.transition_count >= self.thresholds["ascent_stable_frames"]:
                    self.state = "ASCENDING"
                    self.ascent_start_time = frame["timestamp"]
                    for item in self.rep_frames[-self.transition_count:]:
                        item["phase"] = "ASCENDING"
                    self.transition_count = 0

            elif self.state == "ASCENDING":
                self.maximum_return_elbow = max(self.maximum_return_elbow, angle)
                returned = angle >= self.thresholds["top_elbow_angle"]
                self.top_count = self.top_count + 1 if returned else 0
                if self.top_count >= self.thresholds["stable_frames"]:
                    event = self._finish(frame, "full_extension")
                else:
                    reversed_down = (
                        self.maximum_return_elbow >= self.thresholds["incomplete_candidate_angle"]
                        and velocity <= -self.thresholds["descent_velocity_threshold"]
                        and angle < self.thresholds["top_elbow_angle"] - self.thresholds["state_hysteresis"]
                    )
                    self.transition_count = self.transition_count + 1 if reversed_down else 0
                    if self.transition_count >= self.thresholds["stable_frames"]:
                        event = self._finish(frame, "incomplete_extension")

        self.previous_frame = frame
        return {"state": self.state, "phase": self.state, "event": event}
