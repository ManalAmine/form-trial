import os
import subprocess
import sys
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
import sklearn

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    MEDIAPIPE_ERROR = None
except AttributeError as exc:
    mp_drawing = None
    mp_pose = None
    MEDIAPIPE_ERROR = exc

WINDOW_NAME = "Bicep Curl Hybrid Live Prediction"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.getenv(
    "MODEL_FILE",
    os.path.join(SCRIPT_DIR, "bicep_curl_hybrid_quality_model.pkl"),
)

FEATURE_COLUMNS = [
    "min_left_angle",
    "max_left_angle",
    "min_right_angle",
    "max_right_angle",
    "rep_duration",
    "left_rom",
    "right_rom",
    "elbow_rom_diff",
    "concentric_duration",
    "eccentric_duration",
    "left_peak_velocity",
    "right_peak_velocity",
    "torso_lean_mean",
    "torso_lean_max",
    "torso_sway",
    "left_elbow_drift",
    "right_elbow_drift",
    "pose_visibility_mean",
    "pose_visibility_min",
    "tracking_lost_ratio",
]

DOWN_THRESHOLD = 150
START_MOVEMENT_THRESHOLD = 145
MIN_CURL_BEND_THRESHOLD = 130
FULL_CURL_TOP_THRESHOLD = 50
RETURN_THRESHOLD = 145


def get_env_float(name, default_value):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except ValueError:
        print(f"[hybrid-live] Invalid {name}={raw_value!r}; using {default_value:.2f}.")
        return default_value


def get_env_int(name, default_value):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default_value

    try:
        return int(raw_value)
    except ValueError:
        print(f"[hybrid-live] Invalid {name}={raw_value!r}; using {default_value}.")
        return default_value


GOOD_PROBA_THRESHOLD = min(max(get_env_float("GOOD_PROBA_THRESHOLD", 0.50), 0.0), 1.0)
RULE_OVERRIDE_GOOD_PROBA_THRESHOLD = min(
    max(get_env_float("RULE_OVERRIDE_GOOD_PROBA_THRESHOLD", 0.65), 0.0),
    1.0,
)
NO_RULE_GOOD_PROBA_FLOOR = min(
    max(get_env_float("NO_RULE_GOOD_PROBA_FLOOR", 0.28), 0.0),
    1.0,
)
CLEAN_FORM_GOOD_PROBA_FLOOR = min(
    max(get_env_float("CLEAN_FORM_GOOD_PROBA_FLOOR", 0.18), 0.0),
    1.0,
)

MIN_POSE_VISIBILITY_MEAN = 0.65
MIN_POSE_VISIBILITY_MIN = 0.35
MAX_TRACKING_LOST_RATIO = 0.15

MIN_ELBOW_ROM = 35.0
PARTIAL_ROM_MIN_ROM = 50.0
FULL_TOP_ELBOW_ANGLE = 65.0
FULL_BOTTOM_ELBOW_ANGLE = 155.0
CLEAN_FORM_MIN_ELBOW_ROM = 140.0
CLEAN_FORM_MAX_TOP_ELBOW_ANGLE = 20.0
CLEAN_FORM_MIN_REP_DURATION = 0.95

TOO_FAST_REP_DURATION = 1.05
TOO_FAST_CONCENTRIC_DURATION = 0.40
TOO_FAST_ECCENTRIC_DURATION = 0.18
TOO_FAST_MEAN_CONCENTRIC_VELOCITY = 450.0
VERY_FAST_REP_DURATION = 0.90
VERY_FAST_CONCENTRIC_DURATION = 0.28
VERY_FAST_MEAN_CONCENTRIC_VELOCITY = 800.0

MAX_TORSO_SWAY = 8.0
MAX_TORSO_LEAN = 10.0
STRONG_TORSO_LEAN = 10.0
STRONG_TORSO_SWAY = 10.0

MAX_ELBOW_ROM_DIFF = 25.0
MAX_ELBOW_DRIFT = 0.15
MAX_ELBOW_DRIFT_DIFF = 0.08
MAX_ARM_TIMING_DIFF = 1.10
MAX_ARM_TIMING_RATIO = 0.85
MIN_ASYMMETRY_TIMING_ROM_DIFF = 15.0
MIN_REASON_SEVERITY = 0.05

REASON_BASE_SCORES = {
    "partial range of motion": 0.95,
    "too fast": 0.90,
    "torso sway": 0.85,
    "arm asymmetry": 0.70,
}

MIN_PARTIAL_HOLD_FRAMES = 3
MIN_FULL_TOP_HOLD_FRAMES = 5
MIN_RETURN_HOLD_FRAMES = 2
MIN_READY_HOLD_FRAMES = 3
MAX_READY_MISS_FRAMES = 30
MIN_REP_DURATION = 0.45
MAX_REP_DURATION = 6.00
MAX_TOP_WRIST_DISTANCE_RATIO_FULL = 0.95
MAX_TOP_WRIST_DISTANCE_RATIO_PARTIAL = 1.20
MAX_TOP_SHOULDER_ANGLE = 60.0
MAX_SHOULDER_ANGLE_RANGE = 45.0

DISPLAY_REASON_LABELS = {
    "partial range of motion": "partial range of motion",
    "too fast": "too fast",
    "torso sway": "body lean/swing",
    "arm asymmetry": "uneven arm movement",
    "check your form": "check your form",
    "tracking quality": "camera tracking issue",
}

VOICE_REASON_LABELS = {
    "partial range of motion": "use full range of motion",
    "too fast": "slow down",
    "torso sway": "keep your torso upright and still",
    "arm asymmetry": "move both arms evenly",
    "check your form": "check your form",
    "tracking quality": "make sure your arms are visible",
}

BAD_VOICE_FEEDBACK = {
    "partial range of motion": "Bad rep. Use full range of motion.",
    "too fast": "Bad rep. Slow down.",
    "torso sway": "Bad rep. Keep your torso upright and still.",
    "arm asymmetry": "Bad rep. Move both arms evenly.",
    "check your form": "Bad rep. Check your form.",
}

DEBUG_REP_SUMMARY = os.getenv("DEBUG_REP_SUMMARY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
VOICE_FEEDBACK_ENABLED = os.getenv("VOICE_FEEDBACK_ENABLED", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
VOICE_FEEDBACK_RATE = max(min(get_env_int("VOICE_FEEDBACK_RATE", 1), 10), -10)

POSE_SMOOTHING_ALPHA = min(max(get_env_float("POSE_SMOOTHING_ALPHA", 0.35), 0.05), 1.0)
POSE_FAST_SMOOTHING_ALPHA = min(
    max(get_env_float("POSE_FAST_SMOOTHING_ALPHA", 0.78), POSE_SMOOTHING_ALPHA),
    1.0,
)
POSE_JITTER_DEADZONE_PX = max(get_env_float("POSE_JITTER_DEADZONE_PX", 4.0), 0.0)
POSE_FAST_MOVE_PX = max(
    get_env_float("POSE_FAST_MOVE_PX", 28.0),
    POSE_JITTER_DEADZONE_PX + 1.0,
)
POSE_SMOOTHING_RESET_SECONDS = max(
    get_env_float("POSE_SMOOTHING_RESET_SECONDS", 0.35),
    0.0,
)


def calculate_angle(a, b, c):
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    c = np.array(c, dtype=np.float64)

    ba = a - b
    bc = c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 1e-9:
        return 0.0

    cosine_angle = np.clip(np.dot(ba, bc) / denominator, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine_angle)))


def calculate_torso_lean(shoulder_mid, hip_mid):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6))))


def calculate_distance(point_a, point_b):
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


class PoseLandmarkSmoother:
    def __init__(
        self,
        alpha,
        fast_alpha,
        deadzone_px,
        fast_move_px,
        reset_seconds,
    ):
        self.alpha = alpha
        self.fast_alpha = fast_alpha
        self.deadzone_px = deadzone_px
        self.fast_move_px = fast_move_px
        self.reset_seconds = reset_seconds
        self.previous_points = None
        self.previous_time = None

    def reset(self):
        self.previous_points = None
        self.previous_time = None

    def smooth(self, pose_landmarks, frame_width, frame_height, frame_time):
        landmarks = pose_landmarks.landmark
        if (
            self.previous_points is None
            or len(self.previous_points) != len(landmarks)
            or (
                self.previous_time is not None
                and frame_time - self.previous_time > self.reset_seconds
            )
        ):
            self.previous_points = [
                (landmark.x, landmark.y, landmark.z) for landmark in landmarks
            ]
            self.previous_time = frame_time
            return pose_landmarks

        smoothed_points = []
        for landmark, previous in zip(landmarks, self.previous_points):
            raw_x = landmark.x
            raw_y = landmark.y
            raw_z = landmark.z
            distance_px = float(
                np.hypot(
                    (raw_x - previous[0]) * frame_width,
                    (raw_y - previous[1]) * frame_height,
                )
            )

            if distance_px <= self.deadzone_px:
                smooth_x, smooth_y, smooth_z = previous
            else:
                movement_ratio = min(distance_px / self.fast_move_px, 1.0)
                alpha = self.alpha + (self.fast_alpha - self.alpha) * movement_ratio
                smooth_x = previous[0] + alpha * (raw_x - previous[0])
                smooth_y = previous[1] + alpha * (raw_y - previous[1])
                smooth_z = previous[2] + alpha * (raw_z - previous[2])

            landmark.x = smooth_x
            landmark.y = smooth_y
            landmark.z = smooth_z
            smoothed_points.append((smooth_x, smooth_y, smooth_z))

        self.previous_points = smoothed_points
        self.previous_time = frame_time
        return pose_landmarks


def compute_peak_velocity(angles, timestamps):
    if len(angles) < 2 or len(timestamps) < 2:
        return 0.0

    peak_velocity = 0.0
    for prev_angle, curr_angle, prev_time, curr_time in zip(
        angles,
        angles[1:],
        timestamps,
        timestamps[1:],
    ):
        dt = curr_time - prev_time
        if dt <= 1e-6:
            continue
        current_velocity = abs((curr_angle - prev_angle) / dt)
        if current_velocity > peak_velocity:
            peak_velocity = current_velocity

    return float(peak_velocity)


def build_rep_features(
    left_angles,
    right_angles,
    timestamps,
    torso_leans,
    left_elbow_x_values,
    right_elbow_x_values,
    visibility_values,
    rep_start_time,
    rep_total_frames,
    rep_lost_frames,
):
    rep_duration = max(timestamps[-1] - timestamps[0], 0.0) if len(timestamps) >= 2 else 0.0
    left_min = float(min(left_angles))
    left_max = float(max(left_angles))
    right_min = float(min(right_angles))
    right_max = float(max(right_angles))
    left_rom = left_max - left_min
    right_rom = right_max - right_min

    mean_angles = [(left + right) / 2.0 for left, right in zip(left_angles, right_angles)]
    top_index = int(np.argmin(mean_angles))
    left_top_index = int(np.argmin(left_angles))
    right_top_index = int(np.argmin(right_angles))

    concentric_duration = 0.0
    eccentric_duration = 0.0
    arm_timing_diff = 0.0
    arm_timing_ratio = 0.0
    mean_concentric_velocity = 0.0
    if timestamps:
        concentric_duration = max(timestamps[top_index] - timestamps[0], 0.0)
        eccentric_duration = max(timestamps[-1] - timestamps[top_index], 0.0)
        arm_timing_diff = abs(timestamps[left_top_index] - timestamps[right_top_index])
        if concentric_duration > 1e-6:
            left_concentric_rom = max(float(left_angles[0]) - left_min, 0.0)
            right_concentric_rom = max(float(right_angles[0]) - right_min, 0.0)
            mean_concentric_velocity = (
                (left_concentric_rom + right_concentric_rom) / 2.0
            ) / concentric_duration
            arm_timing_ratio = arm_timing_diff / concentric_duration

    torso_mean = float(np.mean(torso_leans)) if torso_leans else 0.0
    torso_max = float(np.max(torso_leans)) if torso_leans else 0.0
    torso_sway = float(np.max(torso_leans) - np.min(torso_leans)) if torso_leans else 0.0

    left_drift = (
        float(max(left_elbow_x_values) - min(left_elbow_x_values))
        if left_elbow_x_values
        else 0.0
    )
    right_drift = (
        float(max(right_elbow_x_values) - min(right_elbow_x_values))
        if right_elbow_x_values
        else 0.0
    )

    visibility_mean = float(np.mean(visibility_values)) if visibility_values else 0.0
    visibility_min = float(np.min(visibility_values)) if visibility_values else 0.0
    tracking_lost_ratio = (
        float(rep_lost_frames) / float(rep_total_frames) if rep_total_frames > 0 else 0.0
    )

    return {
        "min_left_angle": round(left_min, 2),
        "max_left_angle": round(left_max, 2),
        "min_right_angle": round(right_min, 2),
        "max_right_angle": round(right_max, 2),
        "rep_duration": round(rep_duration, 2),
        "left_rom": round(left_rom, 2),
        "right_rom": round(right_rom, 2),
        "elbow_rom_diff": round(abs(left_rom - right_rom), 2),
        "concentric_duration": round(concentric_duration, 2),
        "eccentric_duration": round(eccentric_duration, 2),
        "left_peak_velocity": round(compute_peak_velocity(left_angles, timestamps), 2),
        "right_peak_velocity": round(compute_peak_velocity(right_angles, timestamps), 2),
        "torso_lean_mean": round(torso_mean, 2),
        "torso_lean_max": round(torso_max, 2),
        "torso_sway": round(torso_sway, 2),
        "left_elbow_drift": round(left_drift, 4),
        "right_elbow_drift": round(right_drift, 4),
        "pose_visibility_mean": round(visibility_mean, 4),
        "pose_visibility_min": round(visibility_min, 4),
        "tracking_lost_ratio": round(tracking_lost_ratio, 4),
        "arm_timing_diff": round(float(arm_timing_diff), 3),
        "arm_timing_ratio": round(float(arm_timing_ratio), 3),
        "mean_concentric_velocity": round(float(mean_concentric_velocity), 2),
    }


def build_feature_frame(feature_row, feature_columns):
    ordered_row = {column: feature_row.get(column, 0.0) for column in feature_columns}
    return pd.DataFrame([ordered_row], columns=feature_columns)


def evaluate_tracking_quality(feature_row):
    visibility_mean = float(feature_row.get("pose_visibility_mean", 0.0))
    visibility_min = float(feature_row.get("pose_visibility_min", 0.0))
    lost_ratio = float(feature_row.get("tracking_lost_ratio", 1.0))

    reasons = []
    if visibility_mean < MIN_POSE_VISIBILITY_MEAN:
        reasons.append(f"mean visibility {visibility_mean:.2f} < {MIN_POSE_VISIBILITY_MEAN:.2f}")
    if visibility_min < MIN_POSE_VISIBILITY_MIN:
        reasons.append(f"min visibility {visibility_min:.2f} < {MIN_POSE_VISIBILITY_MIN:.2f}")
    if lost_ratio > MAX_TRACKING_LOST_RATIO:
        reasons.append(f"lost ratio {lost_ratio:.2f} > {MAX_TRACKING_LOST_RATIO:.2f}")

    return len(reasons) == 0, reasons


def evaluate_valid_curl_motion(
    feature_row,
    rep_reached_partial,
    rep_reached_full,
    left_angles,
    right_angles,
    left_wrist_to_shoulder_ratios,
    right_wrist_to_shoulder_ratios,
    left_shoulder_angles,
    right_shoulder_angles,
):
    if (
        not rep_reached_partial
        or not left_angles
        or not right_angles
        or not left_wrist_to_shoulder_ratios
        or not right_wrist_to_shoulder_ratios
        or not left_shoulder_angles
        or not right_shoulder_angles
    ):
        return False, "complete a full curl"

    left_rom = float(feature_row.get("left_rom", 0.0))
    right_rom = float(feature_row.get("right_rom", 0.0))
    if left_rom < MIN_ELBOW_ROM or right_rom < MIN_ELBOW_ROM:
        return False, "complete a full curl"

    top_distance_limit = (
        MAX_TOP_WRIST_DISTANCE_RATIO_FULL
        if rep_reached_full
        else MAX_TOP_WRIST_DISTANCE_RATIO_PARTIAL
    )
    bend_threshold = (
        FULL_CURL_TOP_THRESHOLD
        if rep_reached_full
        else MIN_CURL_BEND_THRESHOLD
    )
    bent_indices = [
        index
        for index, (left_angle, right_angle) in enumerate(zip(left_angles, right_angles))
        if left_angle < bend_threshold and right_angle < bend_threshold
    ]
    if not bent_indices:
        mean_angles = [
            (left_angle + right_angle) / 2.0
            for left_angle, right_angle in zip(left_angles, right_angles)
        ]
        bent_indices = [int(np.argmin(mean_angles))]

    left_top_ratio = float(
        min(left_wrist_to_shoulder_ratios[index] for index in bent_indices)
    )
    right_top_ratio = float(
        min(right_wrist_to_shoulder_ratios[index] for index in bent_indices)
    )
    if (
        rep_reached_full
        and (left_top_ratio > top_distance_limit or right_top_ratio > top_distance_limit)
    ):
        return False, "complete a full curl"

    top_shoulder_angle = max(
        float(max(left_shoulder_angles)),
        float(max(right_shoulder_angles)),
    )
    shoulder_angle_range = max(
        float(max(left_shoulder_angles) - min(left_shoulder_angles)),
        float(max(right_shoulder_angles) - min(right_shoulder_angles)),
    )
    return True, None


def _over_threshold_severity(value, threshold):
    if threshold <= 0.0 or value <= threshold:
        return 0.0
    return min((value - threshold) / threshold, 2.0)


def _under_threshold_severity(value, threshold):
    if threshold <= 0.0 or value >= threshold:
        return 0.0
    return min((threshold - value) / threshold, 2.0)


def get_bicep_curl_reason(feature_row):
    reasons = []

    left_rom = float(feature_row.get("left_rom", 0.0))
    right_rom = float(feature_row.get("right_rom", 0.0))
    min_left = float(feature_row.get("min_left_angle", 180.0))
    min_right = float(feature_row.get("min_right_angle", 180.0))
    max_left = float(feature_row.get("max_left_angle", 0.0))
    max_right = float(feature_row.get("max_right_angle", 0.0))
    rep_duration = float(feature_row.get("rep_duration", 0.0))
    concentric_duration = float(feature_row.get("concentric_duration", 0.0))
    eccentric_duration = float(feature_row.get("eccentric_duration", 0.0))
    torso_sway = float(feature_row.get("torso_sway", 0.0))
    torso_lean_max = float(feature_row.get("torso_lean_max", 0.0))
    elbow_rom_diff = float(feature_row.get("elbow_rom_diff", 0.0))
    left_elbow_drift = float(feature_row.get("left_elbow_drift", 0.0))
    right_elbow_drift = float(feature_row.get("right_elbow_drift", 0.0))
    arm_timing_diff = float(feature_row.get("arm_timing_diff", 0.0))
    arm_timing_ratio = float(feature_row.get("arm_timing_ratio", 0.0))
    mean_concentric_velocity = float(feature_row.get("mean_concentric_velocity", 0.0))
    rep_reached_full = bool(feature_row.get("rep_reached_full", True))

    torso_severity = max(
        _over_threshold_severity(torso_sway, MAX_TORSO_SWAY),
        _over_threshold_severity(torso_lean_max, MAX_TORSO_LEAN),
    )
    strong_torso_issue = (
        torso_sway >= STRONG_TORSO_SWAY
        or torso_lean_max >= STRONG_TORSO_LEAN
    )

    partial_severity = max(
        0.45 if not rep_reached_full else 0.0,
        _under_threshold_severity(left_rom, PARTIAL_ROM_MIN_ROM),
        _under_threshold_severity(right_rom, PARTIAL_ROM_MIN_ROM),
        _over_threshold_severity(min_left, FULL_TOP_ELBOW_ANGLE),
        _over_threshold_severity(min_right, FULL_TOP_ELBOW_ANGLE),
    )
    missing_bottom_extension = (
        max_left < FULL_BOTTOM_ELBOW_ANGLE
        or max_right < FULL_BOTTOM_ELBOW_ANGLE
    )
    if strong_torso_issue:
        return "torso sway"

    if partial_severity >= MIN_REASON_SEVERITY or missing_bottom_extension:
        return "partial range of motion"

    duration_score = 0
    if rep_duration > 0.0:
        if rep_duration < VERY_FAST_REP_DURATION:
            duration_score = 2
        elif rep_duration < TOO_FAST_REP_DURATION:
            duration_score = 1

    concentric_score = 0
    if concentric_duration > 0.0:
        if concentric_duration < VERY_FAST_CONCENTRIC_DURATION:
            concentric_score = 2
        elif concentric_duration < TOO_FAST_CONCENTRIC_DURATION:
            concentric_score = 1

    velocity_score = 0
    if mean_concentric_velocity > VERY_FAST_MEAN_CONCENTRIC_VELOCITY:
        velocity_score = 2
    elif mean_concentric_velocity > TOO_FAST_MEAN_CONCENTRIC_VELOCITY:
        velocity_score = 1

    eccentric_score = 0
    if (
        eccentric_duration > 0.0
        and eccentric_duration < TOO_FAST_ECCENTRIC_DURATION
        and rep_duration < 1.45
    ):
        eccentric_score = 1

    fast_score = duration_score + concentric_score + velocity_score + eccentric_score
    too_fast = (
        duration_score >= 2
        or (duration_score > 0 and duration_score + concentric_score >= 2)
        or duration_score + velocity_score >= 3
        or (concentric_score >= 2 and rep_duration < 1.35)
        or (fast_score >= 3 and rep_duration < 1.45)
    )
    if too_fast:
        reasons.append(
            ("too fast", REASON_BASE_SCORES["too fast"] + fast_score * 0.25)
        )

    if torso_severity >= MIN_REASON_SEVERITY:
        torso_score = REASON_BASE_SCORES["torso sway"] + torso_severity
        reasons.append(("torso sway", torso_score))

    drift_diff = abs(left_elbow_drift - right_elbow_drift)
    drift_severity = 0.0
    if drift_diff > MAX_ELBOW_DRIFT_DIFF:
        drift_severity = max(
            _over_threshold_severity(max(left_elbow_drift, right_elbow_drift), MAX_ELBOW_DRIFT),
            _over_threshold_severity(drift_diff, MAX_ELBOW_DRIFT_DIFF),
        )

    timing_asymmetry_severity = 0.0
    if (
        arm_timing_diff > MAX_ARM_TIMING_DIFF
        and arm_timing_ratio > MAX_ARM_TIMING_RATIO
        and elbow_rom_diff > MIN_ASYMMETRY_TIMING_ROM_DIFF
    ):
        timing_asymmetry_severity = max(
            _over_threshold_severity(arm_timing_diff, MAX_ARM_TIMING_DIFF),
            _over_threshold_severity(arm_timing_ratio, MAX_ARM_TIMING_RATIO),
        )

    asymmetry_severity = max(
        _over_threshold_severity(elbow_rom_diff, MAX_ELBOW_ROM_DIFF),
        drift_severity,
        timing_asymmetry_severity,
    )
    if asymmetry_severity >= MIN_REASON_SEVERITY:
        reasons.append(
            ("arm asymmetry", REASON_BASE_SCORES["arm asymmetry"] + asymmetry_severity)
        )

    if not reasons:
        return "check your form"

    reasons.sort(key=lambda item: item[1], reverse=True)
    return reasons[0][0]


def predict_quality(model, feature_frame):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(feature_frame)[0]
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(f"Quality model classes do not contain positive class 1: {classes}")

        p_good = float(probabilities[classes.index(1)])
        prediction = "GOOD" if p_good >= GOOD_PROBA_THRESHOLD else "BAD"
        return prediction, p_good

    predicted_value = int(model.predict(feature_frame)[0])
    return ("GOOD" if predicted_value == 1 else "BAD"), None


def is_clean_bicep_curl_rep(feature_row):
    if not feature_row:
        return False

    left_rom = float(feature_row.get("left_rom", 0.0))
    right_rom = float(feature_row.get("right_rom", 0.0))
    min_left = float(feature_row.get("min_left_angle", 180.0))
    min_right = float(feature_row.get("min_right_angle", 180.0))
    max_left = float(feature_row.get("max_left_angle", 0.0))
    max_right = float(feature_row.get("max_right_angle", 0.0))
    rep_duration = float(feature_row.get("rep_duration", 0.0))
    elbow_rom_diff = float(feature_row.get("elbow_rom_diff", 0.0))
    torso_sway = float(feature_row.get("torso_sway", 0.0))
    torso_lean_max = float(feature_row.get("torso_lean_max", 0.0))
    left_elbow_drift = float(feature_row.get("left_elbow_drift", 0.0))
    right_elbow_drift = float(feature_row.get("right_elbow_drift", 0.0))
    visibility_mean = float(feature_row.get("pose_visibility_mean", 0.0))
    tracking_lost_ratio = float(feature_row.get("tracking_lost_ratio", 1.0))
    full_top_hold_frames = int(feature_row.get("full_top_hold_frames", 0))
    left_top_ratio = float(
        feature_row.get("min_left_wrist_to_shoulder_ratio", float("inf"))
    )
    right_top_ratio = float(
        feature_row.get("min_right_wrist_to_shoulder_ratio", float("inf"))
    )

    return (
        bool(feature_row.get("rep_reached_full", False))
        and full_top_hold_frames >= MIN_FULL_TOP_HOLD_FRAMES
        and left_rom >= CLEAN_FORM_MIN_ELBOW_ROM
        and right_rom >= CLEAN_FORM_MIN_ELBOW_ROM
        and min_left <= CLEAN_FORM_MAX_TOP_ELBOW_ANGLE
        and min_right <= CLEAN_FORM_MAX_TOP_ELBOW_ANGLE
        and max_left >= FULL_BOTTOM_ELBOW_ANGLE
        and max_right >= FULL_BOTTOM_ELBOW_ANGLE
        and rep_duration >= CLEAN_FORM_MIN_REP_DURATION
        and elbow_rom_diff <= MAX_ELBOW_ROM_DIFF
        and max(left_elbow_drift, right_elbow_drift) <= MAX_ELBOW_DRIFT
        and torso_sway < MAX_TORSO_SWAY
        and torso_lean_max < MAX_TORSO_LEAN
        and visibility_mean >= MIN_POSE_VISIBILITY_MEAN
        and tracking_lost_ratio <= MAX_TRACKING_LOST_RATIO
        and left_top_ratio <= MAX_TOP_WRIST_DISTANCE_RATIO_FULL
        and right_top_ratio <= MAX_TOP_WRIST_DISTANCE_RATIO_FULL
    )


def decide_rep_outcome(model_prediction, good_probability, rule_reason, feature_row=None):
    has_rule_reason = rule_reason != "check your form"

    if good_probability is None:
        if model_prediction == "GOOD":
            return "GOOD", "none"
        return "BAD", rule_reason

    if rule_reason == "partial range of motion":
        return "BAD", rule_reason

    if has_rule_reason and good_probability < RULE_OVERRIDE_GOOD_PROBA_THRESHOLD:
        return "BAD", rule_reason

    if good_probability >= GOOD_PROBA_THRESHOLD:
        return "GOOD", "none"

    if (
        not has_rule_reason
        and good_probability >= CLEAN_FORM_GOOD_PROBA_FLOOR
        and is_clean_bicep_curl_rep(feature_row)
    ):
        return "GOOD", "none"

    if model_prediction == "BAD" and not has_rule_reason:
        return "BAD", "check your form"

    return "BAD", rule_reason if has_rule_reason else "check your form"


def format_rep_debug_summary(
    feature_row,
    good_probability=None,
    tracking_reasons=None,
    invalid_reason=None,
    rule_reason=None,
):
    summary_parts = [
        f"minL={feature_row.get('min_left_angle', 0.0):.1f}",
        f"minR={feature_row.get('min_right_angle', 0.0):.1f}",
        f"maxL={feature_row.get('max_left_angle', 0.0):.1f}",
        f"maxR={feature_row.get('max_right_angle', 0.0):.1f}",
        f"romL={feature_row.get('left_rom', 0.0):.1f}",
        f"romR={feature_row.get('right_rom', 0.0):.1f}",
        f"rep={feature_row.get('rep_duration', 0.0):.2f}s",
        f"conc={feature_row.get('concentric_duration', 0.0):.2f}s",
        f"ecc={feature_row.get('eccentric_duration', 0.0):.2f}s",
        f"torsoSway={feature_row.get('torso_sway', 0.0):.1f}",
        f"torsoLean={feature_row.get('torso_lean_max', 0.0):.1f}",
        f"vis={feature_row.get('pose_visibility_mean', 0.0):.2f}",
        f"lost={feature_row.get('tracking_lost_ratio', 0.0):.2f}",
        f"armTiming={feature_row.get('arm_timing_diff', 0.0):.2f}s",
        f"armTimingRatio={feature_row.get('arm_timing_ratio', 0.0):.2f}",
        f"meanVel={feature_row.get('mean_concentric_velocity', 0.0):.1f}",
        f"fullTop={int(bool(feature_row.get('rep_reached_full', False)))}",
        f"fullTopHold={feature_row.get('full_top_hold_frames', 0)}",
        f"wristRatioL={feature_row.get('min_left_wrist_to_shoulder_ratio', 0.0):.2f}",
        f"wristRatioR={feature_row.get('min_right_wrist_to_shoulder_ratio', 0.0):.2f}",
    ]
    if good_probability is not None:
        summary_parts.append(f"p_good={good_probability:.3f}")
    if tracking_reasons:
        summary_parts.append("tracking=" + "; ".join(tracking_reasons))
    if invalid_reason:
        summary_parts.append(f"invalid={invalid_reason}")
    if rule_reason:
        summary_parts.append(f"rule={rule_reason}")
    return " | ".join(summary_parts)


def get_voice_feedback(prediction, reason):
    if prediction == "GOOD":
        return "Good rep."
    if prediction == "BAD":
        return BAD_VOICE_FEEDBACK.get(reason, "Bad rep. Check your form.")
    if prediction == "RETAKE REP":
        return "Retake rep. Make sure your arms are visible."
    if prediction == "INVALID REP":
        return "Invalid rep. Complete a full curl."
    return None


def speak_text_async(text):
    if not VOICE_FEEDBACK_ENABLED or not text or os.name != "nt":
        return

    escaped_text = text.replace("'", "''")
    powershell_script = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Rate = {VOICE_FEEDBACK_RATE}; "
        f"$speaker.Speak('{escaped_text}')"
    )

    try:
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", powershell_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError as exc:
        print(f"[hybrid-live] Voice feedback unavailable: {exc}")


def load_model_bundle():
    bundle = joblib.load(MODEL_FILE)
    quality_model = bundle.get("quality_model")
    if quality_model is None:
        raise ValueError(
            "Saved bundle does not contain quality_model. "
            "Run train_bicep_hybrid_quality_model.py first."
        )

    feature_columns = bundle.get("feature_columns") or FEATURE_COLUMNS
    missing_features = [column for column in FEATURE_COLUMNS if column not in feature_columns]
    if missing_features:
        raise ValueError(f"Model bundle is missing expected feature columns: {missing_features}")

    runtime_metadata = bundle.get("runtime_metadata", {})
    trained_python = runtime_metadata.get("python_executable")
    trained_sklearn = runtime_metadata.get("sklearn_version")
    current_python = os.path.abspath(sys.executable)
    current_sklearn = sklearn.__version__

    mismatch_messages = []
    if trained_python and os.path.abspath(trained_python).lower() != current_python.lower():
        mismatch_messages.append(
            f"model was trained with {trained_python} but you are running {current_python}"
        )
    if trained_sklearn and trained_sklearn != current_sklearn:
        mismatch_messages.append(
            f"model was trained with scikit-learn {trained_sklearn} but you are running {current_sklearn}"
        )

    if mismatch_messages:
        print("[hybrid-live] Warning: model/runtime environment mismatch detected.")
        for message in mismatch_messages:
            print(f"[hybrid-live] {message}")
        if trained_python:
            print(f"[hybrid-live] Recommended run: {trained_python} predict_bicep_hybrid_live.py")

    return bundle, quality_model, feature_columns


def draw_angle(image, angle, joint, frame_width, frame_height):
    position = tuple(np.multiply(joint, [frame_width, frame_height]).astype(int))
    cv2.putText(
        image,
        str(int(angle)),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_overlay(
    image,
    counter,
    stage,
    prediction_text,
    reason_text,
):
    cv2.rectangle(image, (0, 0), (760, 106), (255, 145, 238), -1)

    cv2.putText(
        image,
        "REPS",
        (24, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(counter),
        (26, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Stage: {stage}",
        (130, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        prediction_text,
        (130, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        reason_text,
        (130, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def append_rep_sample(rep_state, sample, sample_time=None):
    timestamp = sample["frame_time"] if sample_time is None else sample_time

    rep_state["rep_total_frames"] += 1
    rep_state["left_angles_rep"].append(sample["left_angle"])
    rep_state["right_angles_rep"].append(sample["right_angle"])
    rep_state["timestamps_rep"].append(timestamp)
    rep_state["torso_lean_rep"].append(sample["torso_lean"])
    rep_state["left_elbow_x_rep"].append(sample["left_elbow_x"])
    rep_state["right_elbow_x_rep"].append(sample["right_elbow_x"])
    rep_state["visibility_rep"].append(sample["visibility_mean"])
    rep_state["left_wrist_to_shoulder_ratio_rep"].append(
        sample["left_wrist_to_shoulder_ratio"]
    )
    rep_state["right_wrist_to_shoulder_ratio_rep"].append(
        sample["right_wrist_to_shoulder_ratio"]
    )
    rep_state["left_shoulder_angle_rep"].append(sample["left_shoulder_angle"])
    rep_state["right_shoulder_angle_rep"].append(sample["right_shoulder_angle"])


def reset_rep_state():
    return {
        "rep_ready": False,
        "ready_hold_frames": 0,
        "ready_miss_frames": 0,
        "rep_active": False,
        "rep_start_time": None,
        "ready_sample": None,
        "left_angles_rep": [],
        "right_angles_rep": [],
        "timestamps_rep": [],
        "torso_lean_rep": [],
        "left_elbow_x_rep": [],
        "right_elbow_x_rep": [],
        "visibility_rep": [],
        "left_wrist_to_shoulder_ratio_rep": [],
        "right_wrist_to_shoulder_ratio_rep": [],
        "left_shoulder_angle_rep": [],
        "right_shoulder_angle_rep": [],
        "rep_total_frames": 0,
        "rep_lost_frames": 0,
        "rep_reached_partial": False,
        "rep_reached_full": False,
        "partial_hold_frames": 0,
        "full_top_hold_frames": 0,
        "max_full_top_hold_frames": 0,
        "return_hold_frames": 0,
    }


def main():
    if MEDIAPIPE_ERROR is not None:
        raise RuntimeError(
            "This Python interpreter does not have the MediaPipe pose solutions API. "
            "Run with the same interpreter that works for collect_data.py."
        ) from MEDIAPIPE_ERROR

    bundle, model, feature_columns = load_model_bundle()
    print(f"[hybrid-live] Loaded model bundle: {bundle.get('version', 'unknown')}")
    print(f"[hybrid-live] GOOD threshold: {GOOD_PROBA_THRESHOLD:.2f}")
    print(
        "[hybrid-live] Pose smoothing: "
        f"alpha={POSE_SMOOTHING_ALPHA:.2f}, "
        f"fast_alpha={POSE_FAST_SMOOTHING_ALPHA:.2f}, "
        f"deadzone={POSE_JITTER_DEADZONE_PX:.1f}px"
    )

    use_dshow = os.name == "nt" and hasattr(cv2, "CAP_DSHOW")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if use_dshow else cv2.VideoCapture(0)
    if not cap.isOpened() and use_dshow:
        cap.release()
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")

    counter = 0
    stage = "waiting"
    prediction_text = "Prediction: --"
    reason_text = "Reason: none"
    rep_state = reset_rep_state()
    landmark_smoother = PoseLandmarkSmoother(
        POSE_SMOOTHING_ALPHA,
        POSE_FAST_SMOOTHING_ALPHA,
        POSE_JITTER_DEADZONE_PX,
        POSE_FAST_MOVE_PX,
        POSE_SMOOTHING_RESET_SECONDS,
    )

    with mp_pose.Pose(
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame_time = time.time()
            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = pose.process(rgb_image)

            rgb_image.flags.writeable = True
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            frame_height, frame_width = image.shape[:2]

            if results.pose_landmarks:
                landmark_smoother.smooth(
                    results.pose_landmarks,
                    frame_width,
                    frame_height,
                    frame_time,
                )
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                left_elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                left_wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]
                right_shoulder = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y,
                ]
                right_elbow = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y,
                ]
                right_wrist = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y,
                ]
                left_hip = [
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                ]
                right_hip = [
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y,
                ]

                left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_shoulder_angle = calculate_angle(left_hip, left_shoulder, left_elbow)
                right_shoulder_angle = calculate_angle(
                    right_hip,
                    right_shoulder,
                    right_elbow,
                )
                shoulder_mid = [
                    (left_shoulder[0] + right_shoulder[0]) / 2.0,
                    (left_shoulder[1] + right_shoulder[1]) / 2.0,
                ]
                hip_mid = [
                    (left_hip[0] + right_hip[0]) / 2.0,
                    (left_hip[1] + right_hip[1]) / 2.0,
                ]
                torso_lean = calculate_torso_lean(shoulder_mid, hip_mid)
                visibility_mean = float(
                    np.mean(
                        [
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility,
                            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].visibility,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].visibility,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].visibility,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].visibility,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].visibility,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility,
                        ]
                    )
                )

                draw_angle(image, left_angle, left_elbow, frame_width, frame_height)
                draw_angle(image, right_angle, right_elbow, frame_width, frame_height)

                mean_elbow_angle = (left_angle + right_angle) / 2.0
                shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1e-3)
                left_wrist_to_shoulder_ratio = (
                    calculate_distance(left_wrist, left_shoulder) / shoulder_width
                )
                right_wrist_to_shoulder_ratio = (
                    calculate_distance(right_wrist, right_shoulder) / shoulder_width
                )
                arms_down = (
                    mean_elbow_angle > DOWN_THRESHOLD
                    and left_angle > 130
                    and right_angle > 130
                )
                movement_started = (
                    mean_elbow_angle < START_MOVEMENT_THRESHOLD
                    and left_angle < 160
                    and right_angle < 160
                )
                reached_min_bend = (
                    mean_elbow_angle < MIN_CURL_BEND_THRESHOLD
                    and left_angle < 150
                    and right_angle < 150
                )
                reached_full_top = (
                    left_angle < FULL_CURL_TOP_THRESHOLD
                    and right_angle < FULL_CURL_TOP_THRESHOLD
                    and left_wrist_to_shoulder_ratio < MAX_TOP_WRIST_DISTANCE_RATIO_FULL
                    and right_wrist_to_shoulder_ratio < MAX_TOP_WRIST_DISTANCE_RATIO_FULL
                )
                returned_down = (
                    mean_elbow_angle > RETURN_THRESHOLD
                    and left_angle > 125
                    and right_angle > 125
                )
                current_sample = {
                    "frame_time": frame_time,
                    "left_angle": left_angle,
                    "right_angle": right_angle,
                    "torso_lean": torso_lean,
                    "left_elbow_x": left_elbow[0],
                    "right_elbow_x": right_elbow[0],
                    "visibility_mean": visibility_mean,
                    "left_wrist_to_shoulder_ratio": left_wrist_to_shoulder_ratio,
                    "right_wrist_to_shoulder_ratio": right_wrist_to_shoulder_ratio,
                    "left_shoulder_angle": left_shoulder_angle,
                    "right_shoulder_angle": right_shoulder_angle,
                }

                if not rep_state["rep_active"]:
                    if arms_down and visibility_mean >= MIN_POSE_VISIBILITY_MIN:
                        rep_state["ready_hold_frames"] += 1
                        rep_state["ready_miss_frames"] = 0
                        rep_state["ready_sample"] = current_sample

                        if rep_state["ready_hold_frames"] >= MIN_READY_HOLD_FRAMES:
                            rep_state["rep_ready"] = True
                            stage = "ready"
                        else:
                            stage = "arming"
                    else:
                        rep_state["ready_miss_frames"] += 1

                        if rep_state["ready_miss_frames"] >= MAX_READY_MISS_FRAMES:
                            rep_state = reset_rep_state()
                            stage = "waiting"
                        else:
                            if rep_state["rep_ready"]:
                                stage = "ready"
                            elif rep_state["ready_hold_frames"] > 0:
                                stage = "arming"
                            else:
                                stage = "waiting"

                if (
                    (rep_state["rep_ready"] or rep_state["ready_hold_frames"] > 0)
                    and not rep_state["rep_active"]
                    and movement_started
                    and visibility_mean >= MIN_POSE_VISIBILITY_MIN
                ):
                    ready_sample = rep_state.get("ready_sample")
                    rep_state["rep_active"] = True
                    rep_state["rep_start_time"] = frame_time
                    stage = "down"
                    if ready_sample is not None:
                        append_rep_sample(
                            rep_state,
                            ready_sample,
                            sample_time=frame_time,
                        )

                if rep_state["rep_active"]:
                    append_rep_sample(rep_state, current_sample)

                    if reached_min_bend:
                        rep_state["partial_hold_frames"] += 1
                    else:
                        rep_state["partial_hold_frames"] = 0

                    if reached_full_top:
                        rep_state["full_top_hold_frames"] += 1
                        rep_state["max_full_top_hold_frames"] = max(
                            rep_state["max_full_top_hold_frames"],
                            rep_state["full_top_hold_frames"],
                        )
                    else:
                        rep_state["full_top_hold_frames"] = 0

                    if rep_state["partial_hold_frames"] >= MIN_PARTIAL_HOLD_FRAMES:
                        rep_state["rep_reached_partial"] = True
                        stage = "partial"

                    if (
                        rep_state["max_full_top_hold_frames"] >= MIN_FULL_TOP_HOLD_FRAMES
                        and rep_state["rep_reached_partial"]
                        and not rep_state["rep_reached_full"]
                    ):
                        rep_state["rep_reached_full"] = True
                        stage = "up"

                    if (
                        rep_state["rep_reached_partial"]
                        and returned_down
                    ):
                        rep_state["return_hold_frames"] += 1
                    else:
                        rep_state["return_hold_frames"] = 0

                    if (
                        rep_state["rep_reached_partial"]
                        and rep_state["return_hold_frames"] >= MIN_RETURN_HOLD_FRAMES
                    ):
                        feature_row = build_rep_features(
                            rep_state["left_angles_rep"],
                            rep_state["right_angles_rep"],
                            rep_state["timestamps_rep"],
                            rep_state["torso_lean_rep"],
                            rep_state["left_elbow_x_rep"],
                            rep_state["right_elbow_x_rep"],
                            rep_state["visibility_rep"],
                            rep_state["rep_start_time"],
                            rep_state["rep_total_frames"],
                            rep_state["rep_lost_frames"],
                        )
                        feature_row["rep_reached_partial"] = bool(
                            rep_state["rep_reached_partial"]
                        )
                        feature_row["rep_reached_full"] = bool(
                            rep_state["rep_reached_full"]
                        )
                        feature_row["full_top_hold_frames"] = int(
                            rep_state["max_full_top_hold_frames"]
                        )
                        feature_row["min_left_wrist_to_shoulder_ratio"] = round(
                            min(rep_state["left_wrist_to_shoulder_ratio_rep"]),
                            3,
                        )
                        feature_row["min_right_wrist_to_shoulder_ratio"] = round(
                            min(rep_state["right_wrist_to_shoulder_ratio_rep"]),
                            3,
                        )

                        tracking_ok, tracking_reasons = evaluate_tracking_quality(feature_row)
                        if not tracking_ok:
                            prediction = "RETAKE REP"
                            reason = "tracking quality"
                            stage = "retake"
                            prediction_text = f"Prediction: {prediction}"
                            reason_text = f"Reason: {DISPLAY_REASON_LABELS[reason]}"
                            speak_text_async(get_voice_feedback(prediction, reason))
                            if DEBUG_REP_SUMMARY:
                                print(
                                    "[rep-debug] retake | "
                                    + format_rep_debug_summary(
                                        feature_row,
                                        tracking_reasons=tracking_reasons,
                                    )
                                )
                        else:
                            rep_duration = float(feature_row.get("rep_duration", 0.0))
                            invalid_reason = None
                            if rep_duration < MIN_REP_DURATION:
                                invalid_reason = "movement too short"
                            elif rep_duration > MAX_REP_DURATION:
                                invalid_reason = "movement too long"

                            if invalid_reason is None:
                                curl_ok, invalid_reason = evaluate_valid_curl_motion(
                                    feature_row,
                                    rep_state["rep_reached_partial"],
                                    rep_state["rep_reached_full"],
                                    rep_state["left_angles_rep"],
                                    rep_state["right_angles_rep"],
                                    rep_state["left_wrist_to_shoulder_ratio_rep"],
                                    rep_state["right_wrist_to_shoulder_ratio_rep"],
                                    rep_state["left_shoulder_angle_rep"],
                                    rep_state["right_shoulder_angle_rep"],
                                )
                            else:
                                curl_ok = False

                            if not curl_ok:
                                prediction = "INVALID REP"
                                stage = "invalid"
                                prediction_text = f"Prediction: {prediction}"
                                reason_text = f"Reason: {invalid_reason}"
                                speak_text_async(get_voice_feedback(prediction, invalid_reason))
                                if DEBUG_REP_SUMMARY:
                                    print(
                                        "[rep-debug] invalid | "
                                        + format_rep_debug_summary(
                                            feature_row,
                                            invalid_reason=invalid_reason,
                                        )
                                    )
                            else:
                                feature_frame = build_feature_frame(
                                    feature_row,
                                    feature_columns,
                                )
                                model_prediction, good_probability = predict_quality(
                                    model,
                                    feature_frame,
                                )
                                rule_reason = get_bicep_curl_reason(feature_row)
                                prediction, reason = decide_rep_outcome(
                                    model_prediction,
                                    good_probability,
                                    rule_reason,
                                    feature_row,
                                )
                                if prediction == "GOOD":
                                    reason_text = "Reason: none"
                                else:
                                    reason_text = (
                                        "Reason: "
                                        + DISPLAY_REASON_LABELS.get(reason, "check your form")
                                    )

                                counter += 1
                                stage = "predicted"
                                prediction_text = f"Prediction: {prediction}"
                                speak_text_async(get_voice_feedback(prediction, reason))

                                if DEBUG_REP_SUMMARY:
                                    print(
                                        f"[rep-debug] rep={counter} | "
                                        f"model={model_prediction} | "
                                        f"prediction={prediction} | "
                                        + format_rep_debug_summary(
                                            feature_row,
                                            good_probability=good_probability,
                                            rule_reason=rule_reason,
                                        )
                                    )

                        rep_state = reset_rep_state()

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(245, 117, 66), thickness=2, circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=(245, 66, 230), thickness=2, circle_radius=2
                    ),
                )
            else:
                if not rep_state["rep_active"]:
                    if rep_state["rep_ready"] or rep_state["ready_hold_frames"] > 0:
                        rep_state["ready_miss_frames"] += 1
                        if rep_state["ready_miss_frames"] >= MAX_READY_MISS_FRAMES:
                            rep_state = reset_rep_state()
                    if rep_state["rep_ready"]:
                        stage = "ready"
                    elif rep_state["ready_hold_frames"] > 0:
                        stage = "arming"
                    else:
                        stage = "waiting"
                else:
                    rep_state["rep_total_frames"] += 1
                    rep_state["rep_lost_frames"] += 1
                    rep_state["partial_hold_frames"] = 0
                    rep_state["full_top_hold_frames"] = 0
                    rep_state["return_hold_frames"] = 0

            draw_overlay(
                image,
                counter,
                stage,
                prediction_text,
                reason_text,
            )
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
