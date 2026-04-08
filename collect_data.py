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

WINDOW_NAME = "Collect Bicep Curl Data (V2)"
DATASET_FILE = "reps_dataset_v2.csv"
DOWN_THRESHOLD = 160
UP_THRESHOLD = 30
PARTIAL_THRESHOLD = 95
RETURN_THRESHOLD = 125

ACTIVE_ERROR_KEYS = {
    ord("1"): "err_partial_rom",
    ord("2"): "err_too_fast",
    ord("3"): "err_torso_sway",
    ord("4"): "err_asymmetry",
}
ERROR_DISPLAY = {
    "err_partial_rom": "partial",
    "err_too_fast": "fast",
    "err_torso_sway": "torso",
    "err_asymmetry": "asymmetry",
    "err_elbow_drift": "elbow",
    "err_shoulder_swing": "shoulder",
    "err_wrist_compensation": "wrist",
    "err_control_loss": "control",
}
ERROR_COLUMNS = [
    "err_partial_rom",
    "err_too_fast",
    "err_torso_sway",
    "err_elbow_drift",
    "err_asymmetry",
    "err_shoulder_swing",
    "err_wrist_compensation",
    "err_control_loss",
]
ACTIVE_ERROR_COLUMNS = list(ACTIVE_ERROR_KEYS.values())

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
CSV_COLUMNS = FEATURE_COLUMNS + ERROR_COLUMNS + ["is_good"]


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
    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(CSV_COLUMNS)
        return

    with open(filename, mode="r", newline="") as file_handle:
        reader = csv.reader(file_handle)
        existing_header = next(reader, [])

    if existing_header == CSV_COLUMNS:
        return

    backup_filename = f"{filename}.backup_{int(time.time())}"
    os.replace(filename, backup_filename)

    with open(backup_filename, mode="r", newline="") as old_file, open(
        filename, mode="w", newline=""
    ) as new_file:
        reader = csv.DictReader(old_file)
        writer = csv.DictWriter(new_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for row in reader:
            migrated_row = {}
            for column in CSV_COLUMNS:
                value = row.get(column, "")
                if value != "":
                    migrated_row[column] = value
                    continue

                if column in FEATURE_COLUMNS:
                    migrated_row[column] = 0
                elif column in ERROR_COLUMNS:
                    migrated_row[column] = 0
                elif column == "is_good":
                    if row.get("label", "").strip().lower() == "good":
                        migrated_row[column] = 1
                    else:
                        migrated_row[column] = 0
            writer.writerow(migrated_row)


def save_row(filename, row):
    with open(filename, mode="a", newline="") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(row)


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


def format_selected_errors(selected_errors):
    if not selected_errors:
        return "Selected errors: none (good)"

    ordered_labels = [
        ERROR_DISPLAY[column]
        for column in ACTIVE_ERROR_COLUMNS
        if column in selected_errors
    ]
    selected_text = ", ".join(ordered_labels)
    if len(selected_text) > 55:
        selected_text = f"{selected_text[:52]}..."
    return f"Selected errors: {selected_text}"


def draw_overlay(
    image,
    counter,
    stage,
    last_saved_message,
    status_message,
    selected_errors_message,
):
    cv2.rectangle(image, (0, 0), (900, 170), (255, 145, 238), -1)

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
        (130, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        selected_errors_message,
        (130, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        last_saved_message,
        (130, 106),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "Toggle errors: 1-4 | C=clear selected | S=save rep | Q=quit",
        (130, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "1=partial 2=fast 3=torso 4=asym | paused: elbow, shoulder, wrist, control",
        (130, 158),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.41,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )


def build_output_row(feature_row, selected_errors):
    is_good = 1 if not selected_errors else 0

    full_row = dict(feature_row)
    full_row["is_good"] = is_good

    for error_column in ERROR_COLUMNS:
        full_row[error_column] = 1 if error_column in selected_errors else 0

    return [full_row[column] for column in CSV_COLUMNS], is_good


def main():
    ensure_csv_exists(DATASET_FILE)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the default camera.")

    counter = 0
    stage = "waiting"
    last_saved_message = f"Dataset ready: {DATASET_FILE}"
    status_message = "Move into frame with both arms visible."

    rep_active = False
    rep_start_time = None
    left_angles_rep = []
    right_angles_rep = []
    timestamps_rep = []
    torso_lean_rep = []
    left_elbow_x_rep = []
    right_elbow_x_rep = []
    visibility_rep = []
    rep_total_frames = 0
    rep_lost_frames = 0
    rep_reached_partial = False
    rep_reached_full = False
    rep_returned_full = False
    pending_rep = None

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
                arms_up = left_angle < UP_THRESHOLD and right_angle < UP_THRESHOLD

                if pending_rep is None:
                    if arms_down and not rep_active:
                        rep_active = True
                        rep_start_time = frame_time
                        left_angles_rep = []
                        right_angles_rep = []
                        timestamps_rep = []
                        torso_lean_rep = []
                        left_elbow_x_rep = []
                        right_elbow_x_rep = []
                        visibility_rep = []
                        rep_total_frames = 0
                        rep_lost_frames = 0
                        rep_reached_partial = False
                        rep_reached_full = False
                        rep_returned_full = False
                        stage = "down"
                        status_message = (
                            "Rep started. Curl up fully or halfway, then return down."
                        )

                    if rep_active:
                        rep_total_frames += 1
                        left_angles_rep.append(left_angle)
                        right_angles_rep.append(right_angle)
                        timestamps_rep.append(frame_time)
                        torso_lean_rep.append(torso_lean)
                        left_elbow_x_rep.append(left_elbow[0])
                        right_elbow_x_rep.append(right_elbow[0])
                        visibility_rep.append(visibility_mean)

                        if (
                            left_angle < PARTIAL_THRESHOLD
                            and right_angle < PARTIAL_THRESHOLD
                            and not rep_reached_partial
                        ):
                            rep_reached_partial = True
                            stage = "partial"
                            status_message = (
                                "Partial range reached. Return down to finish the rep."
                            )

                        if arms_up and not rep_reached_full:
                            rep_reached_full = True
                            stage = "up"
                            status_message = "Top reached. Lower both arms to finish the rep."
                        elif (
                            left_angle > RETURN_THRESHOLD
                            and right_angle > RETURN_THRESHOLD
                            and rep_reached_partial
                        ):
                            rep_returned_full = arms_down
                            counter += 1

                            features = build_rep_features(
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

                            pending_rep = {
                                "rep_number": counter,
                                "feature_row": features,
                                "selected_errors": set(),
                            }

                            rep_active = False
                            rep_start_time = None
                            left_angles_rep = []
                            right_angles_rep = []
                            timestamps_rep = []
                            torso_lean_rep = []
                            left_elbow_x_rep = []
                            right_elbow_x_rep = []
                            visibility_rep = []
                            rep_total_frames = 0
                            rep_lost_frames = 0
                            rep_reached_partial = False
                            rep_reached_full = False
                            rep_returned_full = False
                            stage = "awaiting labels"
                            status_message = (
                                f"Rep {counter} done. Toggle 1-4, then S (no errors = good)."
                            )
                else:
                    stage = "awaiting labels"
                    status_message = (
                        f"Rep {pending_rep['rep_number']} waiting. S saves good if none selected."
                    )

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
                    if rep_active:
                        rep_total_frames += 1
                        rep_lost_frames += 1
                        status_message = (
                            "Pose lost mid-rep. Step back so both arms are visible."
                        )
                    else:
                        stage = "searching"
                        status_message = "Move into frame so tracking can start."

            selected_errors_message = (
                format_selected_errors(pending_rep["selected_errors"])
                if pending_rep is not None
                else "Selected errors: --"
            )

            draw_overlay(
                image,
                counter,
                stage,
                last_saved_message,
                status_message,
                selected_errors_message,
            )
            cv2.imshow(WINDOW_NAME, image)

            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break

            if pending_rep is not None:
                if key in ACTIVE_ERROR_KEYS:
                    selected_column = ACTIVE_ERROR_KEYS[key]
                    if selected_column in pending_rep["selected_errors"]:
                        pending_rep["selected_errors"].remove(selected_column)
                    else:
                        pending_rep["selected_errors"].add(selected_column)
                elif key == ord("c"):
                    pending_rep["selected_errors"].clear()
                    status_message = "Error selection cleared."
                elif key == ord("s"):
                    row, is_good = build_output_row(
                        pending_rep["feature_row"],
                        pending_rep["selected_errors"],
                    )
                    save_row(DATASET_FILE, row)
                    saved_reasons = (
                        ", ".join(
                            [
                                ERROR_DISPLAY[column]
                                for column in ERROR_COLUMNS
                                if column in pending_rep["selected_errors"]
                            ]
                        )
                        if pending_rep["selected_errors"]
                        else "none"
                    )
                    saved_label = "good" if is_good == 1 else "bad"
                    last_saved_message = (
                        f"Saved rep {pending_rep['rep_number']} as {saved_label} (is_good={is_good}, reasons={saved_reasons})."
                    )
                    status_message = last_saved_message
                    pending_rep = None
                    stage = "ready"
                elif key not in (255,):
                    status_message = (
                        "Use 1-4 to toggle errors, C to clear selected, or S to save."
                    )

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
