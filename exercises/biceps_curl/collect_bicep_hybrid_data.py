import csv
import os
import time

import cv2
import mediapipe as mp
import numpy as np

try:
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
except AttributeError:
    from mediapipe.python.solutions import drawing_utils as mp_drawing
    from mediapipe.python.solutions import pose as mp_pose

WINDOW_NAME = "Collect Bicep Curl Hybrid Data"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.getenv(
    "DATASET_FILE",
    os.path.join(SCRIPT_DIR, "biceps_hybrid_reps_dataset.csv"),
)

DOWN_THRESHOLD = 160
START_MOVEMENT_THRESHOLD = 150
MIN_CURL_BEND_THRESHOLD = 130
FULL_CURL_TOP_THRESHOLD = 65
RETURN_THRESHOLD = 150

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
CSV_COLUMNS = FEATURE_COLUMNS + ["is_good"]


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

    return float(angle)


def calculate_torso_lean(shoulder_mid, hip_mid):
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    return float(np.degrees(np.arctan2(abs(dx), max(abs(dy), 1e-6))))


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


def ensure_csv_exists(filename):
    if os.path.exists(filename):
        return

    with open(filename, mode="w", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(CSV_COLUMNS)


def save_row(filename, feature_row, is_good):
    output_row = dict(feature_row)
    output_row["is_good"] = int(is_good)

    with open(filename, mode="a", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=CSV_COLUMNS)
        writer.writerow(output_row)


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


def draw_overlay(image, counter, stage, status_message, last_saved_message):
    cv2.rectangle(image, (0, 0), (920, 136), (255, 145, 238), -1)
    cv2.putText(
        image,
        "REPS",
        (15, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        str(counter),
        (15, 88),
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
        0.62,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        status_message,
        (130, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        last_saved_message,
        (130, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "After a rep: G=save good | B=save bad | D=discard | Q=quit",
        (130, 116),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (40, 40, 40),
        1,
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


def reset_rep_state():
    return {
        "rep_ready": False,
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
        "rep_total_frames": 0,
        "rep_lost_frames": 0,
        "rep_reached_partial": False,
        "rep_reached_full": False,
    }


def main():
    ensure_csv_exists(DATASET_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")

    counter = 0
    stage = "waiting"
    last_saved_message = f"Dataset ready: {DATASET_FILE}"
    status_message = "Move into frame with both arms visible."
    pending_rep = None
    rep_state = reset_rep_state()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                movement_started = (
                    left_angle < START_MOVEMENT_THRESHOLD
                    and right_angle < START_MOVEMENT_THRESHOLD
                )
                reached_min_bend = (
                    left_angle < MIN_CURL_BEND_THRESHOLD
                    and right_angle < MIN_CURL_BEND_THRESHOLD
                )
                reached_full_top = (
                    left_angle < FULL_CURL_TOP_THRESHOLD
                    and right_angle < FULL_CURL_TOP_THRESHOLD
                )
                current_sample = {
                    "frame_time": frame_time,
                    "left_angle": left_angle,
                    "right_angle": right_angle,
                    "torso_lean": torso_lean,
                    "left_elbow_x": left_elbow[0],
                    "right_elbow_x": right_elbow[0],
                    "visibility_mean": visibility_mean,
                }

                if pending_rep is None:
                    if arms_down and not rep_state["rep_active"]:
                        if not rep_state["rep_ready"]:
                            rep_state = reset_rep_state()
                        rep_state["rep_ready"] = True
                        rep_state["ready_sample"] = current_sample
                        stage = "ready"
                        status_message = "Ready. Start curling to begin the rep timer."

                    if (
                        rep_state["rep_ready"]
                        and not rep_state["rep_active"]
                        and movement_started
                    ):
                        ready_sample = rep_state.get("ready_sample")
                        rep_state["rep_active"] = True
                        rep_state["rep_start_time"] = frame_time
                        stage = "down"
                        status_message = "Rep started. Curl up, then return down."
                        if ready_sample is not None:
                            append_rep_sample(
                                rep_state,
                                ready_sample,
                                sample_time=frame_time,
                            )

                    if rep_state["rep_active"]:
                        append_rep_sample(rep_state, current_sample)

                        if (
                            reached_min_bend
                            and not rep_state["rep_reached_partial"]
                        ):
                            rep_state["rep_reached_partial"] = True
                            stage = "partial"
                            status_message = "Curl bend reached. Return down or curl higher."

                        if reached_full_top and not rep_state["rep_reached_full"]:
                            rep_state["rep_reached_full"] = True
                            stage = "up"
                            status_message = "Top reached. Lower both arms."

                        if (
                            left_angle > RETURN_THRESHOLD
                            and right_angle > RETURN_THRESHOLD
                            and rep_state["rep_reached_partial"]
                        ):
                            counter += 1
                            pending_rep = {
                                "rep_number": counter,
                                "feature_row": build_rep_features(
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
                                ),
                            }
                            rep_state = reset_rep_state()
                            stage = "awaiting label"
                            status_message = f"Rep {counter} done. Press G good, B bad, or D discard."
                else:
                    stage = "awaiting label"

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
                if pending_rep is None:
                    if rep_state["rep_active"]:
                        rep_state["rep_total_frames"] += 1
                        rep_state["rep_lost_frames"] += 1
                        status_message = "Pose lost mid-rep. Step back into frame."
                    elif rep_state["rep_ready"]:
                        stage = "ready"
                        status_message = "Ready. Start curling to begin the rep timer."
                    else:
                        stage = "searching"
                        status_message = "Move into frame so tracking can start."

            draw_overlay(image, counter, stage, status_message, last_saved_message)
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break

            if pending_rep is not None:
                if key == ord("g"):
                    save_row(DATASET_FILE, pending_rep["feature_row"], is_good=1)
                    last_saved_message = f"Saved rep {pending_rep['rep_number']} as GOOD."
                    status_message = last_saved_message
                    pending_rep = None
                    stage = "ready"
                elif key == ord("b"):
                    save_row(DATASET_FILE, pending_rep["feature_row"], is_good=0)
                    last_saved_message = f"Saved rep {pending_rep['rep_number']} as BAD."
                    status_message = last_saved_message
                    pending_rep = None
                    stage = "ready"
                elif key == ord("d"):
                    last_saved_message = f"Discarded rep {pending_rep['rep_number']}."
                    status_message = last_saved_message
                    pending_rep = None
                    stage = "ready"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
