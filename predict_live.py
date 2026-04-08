import os
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    MEDIAPIPE_ERROR = None
except AttributeError as exc:
    mp_drawing = None
    mp_pose = None
    MEDIAPIPE_ERROR = exc

WINDOW_NAME = "Bicep Curl Live Prediction"
MODEL_FILE = "bicep_curl_model.pkl"

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

ERROR_LABELS = {
    "err_partial_rom": "partial",
    "err_too_fast": "fast",
    "err_torso_sway": "torso",
    "err_asymmetry": "asymmetry",
}
ACTIVE_REASON_KEYS = list(ERROR_LABELS.keys())

DOWN_THRESHOLD = 160
UP_THRESHOLD = 30
PARTIAL_THRESHOLD = 95
RETURN_THRESHOLD = 125

def get_env_float(name, default_value):
    raw_value = os.getenv(name)
    if raw_value is None:
        return default_value

    try:
        return float(raw_value)
    except ValueError:
        print(
            f"[predict_live] Invalid {name}={raw_value!r}; using default {default_value:.2f}."
        )
        return default_value


# Predict GOOD only when model confidence is at/above this threshold.
# With the current small dataset, 0.50 is less punitive than 0.60.
GOOD_PROBA_THRESHOLD = min(max(get_env_float("GOOD_PROBA_THRESHOLD", 0.50), 0.0), 1.0)
ERROR_PROBA_THRESHOLD = min(max(get_env_float("ERROR_PROBA_THRESHOLD", 0.45), 0.0), 1.0)
DEBUG_REP_SUMMARY = os.getenv("DEBUG_REP_SUMMARY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}

# Tracking quality gates. If a rep fails any gate, we ask for RETAKE REP.
MIN_POSE_VISIBILITY_MEAN = 0.65
MIN_POSE_VISIBILITY_MIN = 0.35
MAX_TRACKING_LOST_RATIO = 0.15

# Curl definition. Keep this focused on the core motion: real elbow flexion with
# stable shoulders that holds briefly and then returns down.
MIN_ELBOW_ROM = 35.0
MIN_PARTIAL_HOLD_FRAMES = 3
MIN_RETURN_HOLD_FRAMES = 2
MAX_TOP_WRIST_DISTANCE_RATIO_FULL = 0.95
MAX_TOP_WRIST_DISTANCE_RATIO_PARTIAL = 1.20
MAX_TOP_SHOULDER_ANGLE = 60.0
MAX_SHOULDER_ANGLE_RANGE = 45.0


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def calculate_torso_lean(shoulder_mid, hip_mid):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6))))


def calculate_distance(point_a, point_b):
    return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))


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
    rep_duration = max(time.time() - rep_start_time, 0.0)
    left_min = float(min(left_angles))
    left_max = float(max(left_angles))
    right_min = float(min(right_angles))
    right_max = float(max(right_angles))
    left_rom = left_max - left_min
    right_rom = right_max - right_min

    mean_angles = [(left + right) / 2.0 for left, right in zip(left_angles, right_angles)]
    top_index = int(np.argmin(mean_angles))

    concentric_duration = 0.0
    eccentric_duration = 0.0
    if timestamps:
        concentric_duration = max(timestamps[top_index] - timestamps[0], 0.0)
        eccentric_duration = max(timestamps[-1] - timestamps[top_index], 0.0)

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
    }


def build_feature_frame(feature_row, feature_columns):
    ordered_row = {column: feature_row.get(column, 0.0) for column in feature_columns}
    return pd.DataFrame([ordered_row], columns=feature_columns)


def format_rep_debug_summary(feature_row, good_probability=None, tracking_reasons=None):
    summary_parts = [
        f"minL={feature_row.get('min_left_angle', 0.0):.1f}",
        f"minR={feature_row.get('min_right_angle', 0.0):.1f}",
        f"romL={feature_row.get('left_rom', 0.0):.1f}",
        f"romR={feature_row.get('right_rom', 0.0):.1f}",
        f"rep={feature_row.get('rep_duration', 0.0):.2f}s",
        f"conc={feature_row.get('concentric_duration', 0.0):.2f}s",
        f"ecc={feature_row.get('eccentric_duration', 0.0):.2f}s",
        f"vis={feature_row.get('pose_visibility_mean', 0.0):.2f}",
        f"lost={feature_row.get('tracking_lost_ratio', 0.0):.2f}",
    ]
    if good_probability is not None:
        summary_parts.append(f"p_good={good_probability:.3f}")
    if tracking_reasons:
        summary_parts.append("tracking=" + "; ".join(tracking_reasons))
    return " | ".join(summary_parts)


def predict_error_reasons(error_models, feature_frame):
    reason_predictions = []

    for error_key in ACTIVE_REASON_KEYS:
        meta = error_models.get(error_key)
        if meta is None:
            continue

        probability = 0.0

        if meta.get("type") == "constant":
            probability = float(meta.get("value", 0))
        else:
            error_model = meta.get("model")
            if error_model is None:
                continue

            if hasattr(error_model, "predict_proba"):
                probabilities = error_model.predict_proba(feature_frame)[0]
                classes = list(error_model.classes_)
                if 1 in classes:
                    probability = float(probabilities[classes.index(1)])
                else:
                    probability = float(np.max(probabilities))
            else:
                probability = float(error_model.predict(feature_frame)[0])

        reason_predictions.append(
            {
                "key": error_key,
                "label": ERROR_LABELS.get(error_key, error_key),
                "probability": probability,
            }
        )

    reason_predictions.sort(key=lambda item: item["probability"], reverse=True)
    return reason_predictions


def format_reason_text(predicted_value, reason_predictions):
    if predicted_value == 1:
        return "Reason: none"

    strong_reasons = [
        reason
        for reason in reason_predictions
        if reason["probability"] >= ERROR_PROBA_THRESHOLD
    ]

    if strong_reasons:
        return f"Reason: {strong_reasons[0]['label']}"

    if reason_predictions:
        top_reason = reason_predictions[0]
        return f"Reason: {top_reason['label']}"

    return "Reason: none"


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


def evaluate_curl_motion(
    rep_reached_full,
    left_angles,
    right_angles,
    left_wrist_to_shoulder_ratios,
    right_wrist_to_shoulder_ratios,
    left_shoulder_angles,
    right_shoulder_angles,
):
    if (
        not left_angles
        or not right_angles
        or not left_wrist_to_shoulder_ratios
        or not right_wrist_to_shoulder_ratios
        or not left_shoulder_angles
        or not right_shoulder_angles
    ):
        return False, ["missing wrist path"]

    reasons = []
    left_rom = float(max(left_angles) - min(left_angles))
    right_rom = float(max(right_angles) - min(right_angles))
    if left_rom < MIN_ELBOW_ROM or right_rom < MIN_ELBOW_ROM:
        reasons.append("not enough elbow bend")

    top_distance_limit = (
        MAX_TOP_WRIST_DISTANCE_RATIO_FULL
        if rep_reached_full
        else MAX_TOP_WRIST_DISTANCE_RATIO_PARTIAL
    )
    bent_indices = [
        index
        for index, (left_angle, right_angle) in enumerate(zip(left_angles, right_angles))
        if left_angle < PARTIAL_THRESHOLD and right_angle < PARTIAL_THRESHOLD
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

    if left_top_ratio > top_distance_limit or right_top_ratio > top_distance_limit:
        reasons.append("wrists never came near shoulders")

    top_shoulder_angle = max(
        float(max(left_shoulder_angles)),
        float(max(right_shoulder_angles)),
    )
    shoulder_angle_range = max(
        float(max(left_shoulder_angles) - min(left_shoulder_angles)),
        float(max(right_shoulder_angles) - min(right_shoulder_angles)),
    )
    if top_shoulder_angle > MAX_TOP_SHOULDER_ANGLE:
        reasons.append("shoulders lifting too much")
    if shoulder_angle_range > MAX_SHOULDER_ANGLE_RANGE:
        reasons.append("upper arms moving too much")

    return len(reasons) == 0, reasons


def load_model_bundle():
    bundle = joblib.load(MODEL_FILE)
    quality_model = bundle.get("quality_model")
    if quality_model is None:
        raise ValueError(
            "Saved bundle does not contain a trained quality model. "
            "Train the notebook again with both classes in is_good."
        )

    feature_columns = bundle.get("feature_columns")
    if not feature_columns:
        feature_columns = FEATURE_COLUMNS

    error_models = bundle.get("error_models", {})

    return bundle, quality_model, feature_columns, error_models


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


def main():
    if MEDIAPIPE_ERROR is not None:
        raise RuntimeError(
            "This Python interpreter does not have the MediaPipe pose solutions API. "
            "Run predict_live.py with the same interpreter that works for collect_data.py."
        ) from MEDIAPIPE_ERROR

    bundle, model, feature_columns, error_models = load_model_bundle()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")

    counter = 0
    stage = "waiting"
    prediction_text = "Prediction: --"
    reason_text = "Reason: --"

    rep_active = False
    rep_start_time = None
    left_angles_rep = []
    right_angles_rep = []
    timestamps_rep = []
    torso_lean_rep = []
    left_elbow_x_rep = []
    right_elbow_x_rep = []
    visibility_rep = []
    left_wrist_to_shoulder_ratio_rep = []
    right_wrist_to_shoulder_ratio_rep = []
    left_shoulder_angle_rep = []
    right_shoulder_angle_rep = []
    rep_total_frames = 0
    rep_lost_frames = 0
    rep_reached_partial = False
    rep_reached_full = False
    partial_hold_frames = 0
    return_hold_frames = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_image.flags.writeable = False
            results = pose.process(rgb_image)

            rgb_image.flags.writeable = True
            image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            frame_height, frame_width = image.shape[:2]

            if results.pose_landmarks:
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

                arms_down = left_angle > DOWN_THRESHOLD and right_angle > DOWN_THRESHOLD
                arms_up = left_angle < UP_THRESHOLD and right_angle < UP_THRESHOLD
                shoulder_width = max(abs(left_shoulder[0] - right_shoulder[0]), 1e-3)
                left_wrist_to_shoulder_ratio = (
                    calculate_distance(left_wrist, left_shoulder) / shoulder_width
                )
                right_wrist_to_shoulder_ratio = (
                    calculate_distance(right_wrist, right_shoulder) / shoulder_width
                )

                if arms_down and not rep_active:
                    rep_active = True
                    rep_start_time = time.time()
                    left_angles_rep = []
                    right_angles_rep = []
                    timestamps_rep = []
                    torso_lean_rep = []
                    left_elbow_x_rep = []
                    right_elbow_x_rep = []
                    visibility_rep = []
                    left_wrist_to_shoulder_ratio_rep = []
                    right_wrist_to_shoulder_ratio_rep = []
                    left_shoulder_angle_rep = []
                    right_shoulder_angle_rep = []
                    rep_total_frames = 0
                    rep_lost_frames = 0
                    rep_reached_partial = False
                    rep_reached_full = False
                    partial_hold_frames = 0
                    return_hold_frames = 0
                    stage = "down"

                if rep_active:
                    rep_total_frames += 1
                    left_angles_rep.append(left_angle)
                    right_angles_rep.append(right_angle)
                    timestamps_rep.append(time.time())
                    torso_lean_rep.append(torso_lean)
                    left_elbow_x_rep.append(left_elbow[0])
                    right_elbow_x_rep.append(right_elbow[0])
                    visibility_rep.append(visibility_mean)
                    left_wrist_to_shoulder_ratio_rep.append(left_wrist_to_shoulder_ratio)
                    right_wrist_to_shoulder_ratio_rep.append(right_wrist_to_shoulder_ratio)
                    left_shoulder_angle_rep.append(left_shoulder_angle)
                    right_shoulder_angle_rep.append(right_shoulder_angle)

                    if left_angle < PARTIAL_THRESHOLD and right_angle < PARTIAL_THRESHOLD:
                        partial_hold_frames += 1
                    else:
                        partial_hold_frames = 0

                    if partial_hold_frames >= MIN_PARTIAL_HOLD_FRAMES:
                        rep_reached_partial = True
                        stage = "partial"

                    if arms_up and rep_reached_partial and not rep_reached_full:
                        rep_reached_full = True
                        stage = "up"

                    if (
                        rep_reached_partial
                        and left_angle > RETURN_THRESHOLD
                        and right_angle > RETURN_THRESHOLD
                    ):
                        return_hold_frames += 1
                    else:
                        return_hold_frames = 0

                    if rep_reached_partial and return_hold_frames >= MIN_RETURN_HOLD_FRAMES:
                        feature_row = build_rep_features(
                            left_angles_rep,
                            right_angles_rep,
                            timestamps_rep,
                            torso_lean_rep,
                            left_elbow_x_rep,
                            right_elbow_x_rep,
                            visibility_rep,
                            rep_start_time,
                            rep_total_frames,
                            rep_lost_frames,
                        )
                        tracking_ok, tracking_reasons = evaluate_tracking_quality(feature_row)
                        if not tracking_ok:
                            if DEBUG_REP_SUMMARY:
                                print(
                                    "[rep-debug] retake | "
                                    + format_rep_debug_summary(
                                        feature_row,
                                        tracking_reasons=tracking_reasons,
                                    )
                                )
                            stage = "retake"
                            prediction_text = "Prediction: RETAKE REP"
                            reason_text = "Reason: tracking quality"
                        else:
                            curl_ok, curl_reasons = evaluate_curl_motion(
                                rep_reached_full,
                                left_angles_rep,
                                right_angles_rep,
                                left_wrist_to_shoulder_ratio_rep,
                                right_wrist_to_shoulder_ratio_rep,
                                left_shoulder_angle_rep,
                                right_shoulder_angle_rep,
                            )
                            if not curl_ok:
                                prediction_text = "Prediction: INVALID REP"
                                reason_text = f"Reason: {curl_reasons[0]}"
                                if DEBUG_REP_SUMMARY:
                                    print(
                                        "[rep-debug] invalid | "
                                        + format_rep_debug_summary(feature_row)
                                    )
                                    print(
                                        "[rep-debug] invalid-reasons | "
                                        + ", ".join(curl_reasons)
                                    )
                                stage = "invalid"
                            else:
                                feature_frame = build_feature_frame(feature_row, feature_columns)
                                good_probability = None
                                reason_predictions = []

                                if hasattr(model, "predict_proba"):
                                    probabilities = model.predict_proba(feature_frame)[0]
                                    classes = list(model.classes_)
                                    if 1 in classes:
                                        good_probability = float(
                                            probabilities[classes.index(1)]
                                        )
                                        predicted_value = (
                                            1
                                            if good_probability >= GOOD_PROBA_THRESHOLD
                                            else 0
                                        )
                                    else:
                                        predicted_value = int(model.predict(feature_frame)[0])
                                else:
                                    predicted_value = int(model.predict(feature_frame)[0])

                                predicted_text_value = "GOOD" if predicted_value == 1 else "BAD"
                                prediction_text = f"Prediction: {predicted_text_value}"
                                reason_predictions = predict_error_reasons(
                                    error_models,
                                    feature_frame,
                                )
                                reason_text = format_reason_text(
                                    predicted_value,
                                    reason_predictions,
                                )

                                counter += 1
                                if DEBUG_REP_SUMMARY:
                                    print(
                                        f"[rep-debug] rep={counter} | "
                                        f"prediction={predicted_text_value} | "
                                        + format_rep_debug_summary(
                                            feature_row,
                                            good_probability=good_probability,
                                        )
                                    )
                                    if reason_predictions:
                                        print(
                                            "[rep-debug] reasons | "
                                            + ", ".join(
                                                [
                                                    f"{item['label']}={item['probability']:.3f}"
                                                    for item in reason_predictions[:4]
                                                ]
                                            )
                                        )
                                stage = "predicted"
                        rep_active = False
                        rep_start_time = None
                        left_angles_rep = []
                        right_angles_rep = []
                        timestamps_rep = []
                        torso_lean_rep = []
                        left_elbow_x_rep = []
                        right_elbow_x_rep = []
                        visibility_rep = []
                        left_wrist_to_shoulder_ratio_rep = []
                        right_wrist_to_shoulder_ratio_rep = []
                        left_shoulder_angle_rep = []
                        right_shoulder_angle_rep = []
                        rep_total_frames = 0
                        rep_lost_frames = 0
                        rep_reached_partial = False
                        rep_reached_full = False
                        partial_hold_frames = 0
                        return_hold_frames = 0

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
                if not rep_active:
                    stage = "searching"
                else:
                    rep_total_frames += 1
                    rep_lost_frames += 1
                    partial_hold_frames = 0
                    return_hold_frames = 0

            draw_overlay(
                image,
                counter,
                stage,
                prediction_text,
                reason_text,
            )
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
