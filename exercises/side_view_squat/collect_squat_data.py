"""Interactive webcam/video collection of one CSV row per completed squat rep."""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    from .config import ERROR_COLUMNS
    from .squat_camera import assess_camera
    from .squat_collection import append_sample, ensure_dataset, sample_counts, undo_last_sample
    from .squat_features import build_rep_features
    from .squat_rep_tracker import SquatRepTracker
except ImportError:
    from config import ERROR_COLUMNS
    from squat_camera import assess_camera
    from squat_collection import append_sample, ensure_dataset, sample_counts, undo_last_sample
    from squat_features import build_rep_features
    from squat_rep_tracker import SquatRepTracker


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "data" / "side_squat_reps.csv"
WINDOW_NAME = "VITAL-PT Side Squat Data Collection"
ERROR_KEYS = {
    ord("1"): "err_shallow_depth",
    ord("2"): "err_incomplete_lockout",
    ord("3"): "err_heel_lift",
    ord("4"): "err_excessive_torso_lean",
    ord("5"): "err_chest_collapse",
    ord("6"): "err_uncontrolled_tempo",
}
ERROR_SHORT = {
    "err_shallow_depth": "shallow",
    "err_incomplete_lockout": "not standing",
    "err_heel_lift": "heel lift",
    "err_excessive_torso_lean": "torso lean",
    "err_chest_collapse": "chest collapse",
    "err_uncontrolled_tempo": "uncontrolled tempo",
}


class MouseControls:
    """Small OpenCV mouse-button dispatcher whose rectangles follow the frame."""

    def __init__(self):
        self.buttons: dict[str, tuple[int, int, int, int]] = {}
        self.action: str | None = None

    def set_buttons(self, buttons: dict[str, tuple[int, int, int, int]]) -> None:
        self.buttons = buttons

    def callback(self, event, x, y, _flags, _parameter) -> None:
        if event != cv2.EVENT_LBUTTONUP:
            return
        for action, (left, top, right, bottom) in self.buttons.items():
            if left <= x <= right and top <= y <= bottom:
                self.action = action
                return

    def consume(self) -> str | None:
        action, self.action = self.action, None
        return action


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="Camera index or video path.")
    parser.add_argument("--dataset", default=os.getenv("SQUAT_DATASET", str(DEFAULT_DATASET)))
    parser.add_argument("--participant-id", default=os.getenv("PARTICIPANT_ID", "p001"))
    parser.add_argument("--session-id", default=os.getenv("SESSION_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--recording-id", default=os.getenv("RECORDING_ID", ""))
    parser.add_argument(
        "--collect-reasons",
        action="store_true",
        help="After BAD, optionally record one manual reason for rule evaluation. The ML model never trains on it.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser


def _capture_source(value: str):
    return int(value) if value.isdigit() else value


def _put(
    image, text: str, y: int, color=(255, 255, 255), scale=0.55,
    x=12, max_width=None, thickness=2,
):
    available = max_width or max(image.shape[1] - x - 12, 80)
    fitted_scale = scale
    while fitted_scale > 0.32:
        width = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, fitted_scale, thickness
        )[0][0]
        if width <= available:
            break
        fitted_scale -= 0.04
    cv2.putText(
        image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        fitted_scale, color, thickness, cv2.LINE_AA,
    )


def _draw_button(image, rect, label, color):
    left, top, right, bottom = rect
    cv2.rectangle(image, (left, top), (right, bottom), color, -1)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), 2)
    scale = 0.70
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    while scale > 0.34 and text_size[0] > right - left - 12:
        scale -= 0.04
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
    text_x = left + max((right - left - text_size[0]) // 2, 4)
    text_y = top + max((bottom - top + text_size[1]) // 2, text_size[1] + 2)
    cv2.putText(
        image, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        scale, (255, 255, 255), 2, cv2.LINE_AA,
    )


def _draw_overlay(
    image, tracker, camera_text, pending, selected, good, bad, status,
    measurement=None, controls=None, dataset_name="side_squat_reps.csv",
):
    # Draw into a fixed-size canvas. WINDOW_AUTOSIZE then keeps mouse coordinates
    # identical to these button rectangles on every camera resolution.
    canvas = np.full((720, 1200, 3), (25, 25, 28), dtype=np.uint8)
    source_height, source_width = image.shape[:2]
    camera_left, camera_top = 370, 82
    camera_width, camera_height = 810, 610
    resize_scale = min(camera_width / source_width, camera_height / source_height)
    resized_width = max(1, int(source_width * resize_scale))
    resized_height = max(1, int(source_height * resize_scale))
    resized = cv2.resize(image, (resized_width, resized_height))
    offset_x = camera_left + (camera_width - resized_width) // 2
    offset_y = camera_top + (camera_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    cv2.rectangle(
        canvas,
        (camera_left - 2, camera_top - 2),
        (camera_left + camera_width + 2, camera_top + camera_height + 2),
        (150, 150, 150),
        2,
    )

    phase_messages = {
        "CALIBRATING": ("CALIBRATING", "Stand tall and keep still", (40, 150, 220)),
        "STANDING": ("READY", "Start your squat", (40, 170, 70)),
        "DESCENDING": ("REP IN PROGRESS", "Keep moving down", (50, 150, 220)),
        "BOTTOM": ("BOTTOM FOUND", "Now stand back up", (50, 150, 220)),
        "ASCENDING": ("REP IN PROGRESS", "Stand tall and hold", (50, 150, 220)),
    }
    banner, instruction, banner_color = phase_messages.get(
        tracker.phase, (tracker.phase, "", (70, 70, 70))
    )
    if pending:
        banner, instruction, banner_color = (
            "REP CAPTURED!", "Choose GOOD, BAD, or DISCARD", (35, 150, 55)
        )
    cv2.rectangle(canvas, (0, 0), (1200, 70), banner_color, -1)
    _put(canvas, banner, 34, (255, 255, 255), 0.92, 20, 330, 2)
    _put(canvas, instruction, 35, (255, 255, 255), 0.72, 390, 770, 2)

    panel_x, panel_width = 18, 330
    calibration_text = ""
    if tracker.phase == "CALIBRATING":
        calibration_text = f"Calibration: {len(tracker.calibration)}/{tracker.calibration.maxlen}"
        _put(canvas, calibration_text, 108, (120, 230, 255), 0.66, panel_x, panel_width)
    else:
        _put(canvas, f"Phase: {tracker.phase}", 108, (255, 235, 120), 0.66, panel_x, panel_width)
    if measurement:
        _put(
            canvas,
            f"Knee angle: {measurement['knee_angle']:.0f} deg",
            140,
            (255, 255, 255),
            0.58,
            panel_x,
            panel_width,
        )
        _put(canvas, f"Hip angle: {measurement['hip_angle']:.0f} deg", 169, (255, 255, 255), 0.58, panel_x, panel_width)
        _put(canvas, f"Tracked side: {measurement['side']}", 198, (210, 210, 210), 0.50, panel_x, panel_width)
    else:
        _put(canvas, "Pose angles: not available", 140, (180, 180, 180), 0.52, panel_x, panel_width)

    if tracker.baseline and measurement and not pending:
        bend = max(tracker.baseline["knee_angle"] - measurement["knee_angle"], 0.0)
        _put(canvas, f"Current bend: {bend:.0f} deg", 227, (210, 210, 210), 0.50, panel_x, panel_width)

    if pending:
        stage = pending.get("label_stage", "quality")
        if stage == "quality":
            _put(canvas, "HOW WAS THIS REP?", 270, (255, 255, 255), 0.68, panel_x, panel_width)
            good_button = (18, 290, 348, 370)
            bad_button = (18, 385, 348, 465)
            discard_button = (18, 480, 348, 540)
            _draw_button(canvas, good_button, "GOOD FORM", (35, 150, 55))
            _draw_button(canvas, bad_button, "BAD FORM", (45, 45, 190))
            _draw_button(canvas, discard_button, "DISCARD / NOT SURE", (95, 95, 95))
            if pending.get("collect_reasons", False):
                _put(canvas, "BAD opens optional notes.", 568, (220, 220, 220), 0.45, panel_x, panel_width)
            else:
                _put(canvas, "GOOD or BAD saves immediately.", 568, (220, 220, 220), 0.45, panel_x, panel_width)
            if controls:
                controls.set_buttons({
                    "good": good_button,
                    "bad_menu": bad_button,
                    "discard": discard_button,
                })
        else:
            _put(canvas, "WHY WAS IT BAD?", 258, (255, 255, 255), 0.64, panel_x, panel_width)
            gap = 6
            middle = 183
            left = (18, middle - gap)
            right = (middle + gap, 348)
            reason_buttons = {
                "reason:err_shallow_depth": (left[0], 275, left[1], 325),
                "reason:err_incomplete_lockout": (right[0], 275, right[1], 325),
                "reason:err_heel_lift": (left[0], 335, left[1], 385),
                "reason:err_excessive_torso_lean": (right[0], 335, right[1], 385),
                "reason:err_chest_collapse": (left[0], 395, left[1], 445),
                "reason:err_uncontrolled_tempo": (right[0], 395, right[1], 445),
            }
            reason_labels = {
                "reason:err_shallow_depth": "NOT LOW ENOUGH",
                "reason:err_incomplete_lockout": "DID NOT STAND TALL",
                "reason:err_heel_lift": "HEEL CAME UP",
                "reason:err_excessive_torso_lean": "LEANED TOO FAR",
                "reason:err_chest_collapse": "HIPS ROSE FIRST",
                "reason:err_uncontrolled_tempo": "MOVED TOO FAST",
            }
            for action, rect in reason_buttons.items():
                _draw_button(canvas, rect, reason_labels[action], (55, 80, 180))
            unsure_button = (18, 460, 177, 515)
            back_button = (189, 460, 348, 515)
            _draw_button(canvas, unsure_button, "REASON NOT SURE", (105, 75, 145))
            _draw_button(canvas, back_button, "BACK", (95, 95, 95))
            if controls:
                controls.set_buttons({
                    **reason_buttons,
                    "bad_unsure": unsure_button,
                    "back": back_button,
                })
    else:
        if controls:
            controls.set_buttons({})
        _put(canvas, "HOW CAPTURE WORKS", 285, (255, 255, 255), 0.62, panel_x, panel_width)
        _put(canvas, "1. Stand tall", 322, (220, 220, 220), 0.55, panel_x, panel_width)
        _put(canvas, "2. Squat down", 353, (220, 220, 220), 0.55, panel_x, panel_width)
        _put(canvas, "3. Stand tall and hold", 384, (220, 220, 220), 0.55, panel_x, panel_width)
        _put(canvas, "Then REP CAPTURED appears.", 425, (180, 255, 180), 0.52, panel_x, panel_width)

    _put(canvas, f"Detected reps: {tracker.rep_number}", 610, (255, 255, 255), 0.54, panel_x, panel_width)
    _put(canvas, f"CSV rows: GOOD {good}  BAD {bad}", 640, (255, 255, 255), 0.54, panel_x, panel_width)
    _put(canvas, f"CSV file: {dataset_name}", 668, (190, 190, 190), 0.43, panel_x, panel_width)
    _put(canvas, "Q quit | U undo last save", 698, (190, 190, 190), 0.43, panel_x, panel_width)

    guidance_color = (120, 230, 255) if camera_text == "Camera position is good." else (90, 180, 255)
    cv2.rectangle(canvas, (370, 675), (1180, 714), (35, 35, 38), -1)
    _put(canvas, f"Camera: {camera_text}", 704, guidance_color, 0.56, 382, 780)
    if status:
        _put(canvas, f"Status: {status}", 65, (240, 240, 240), 0.42, 20, 330, 1)
    return canvas


def run(args) -> None:
    dataset = ensure_dataset(args.dataset)
    good_count, bad_count = sample_counts(dataset)
    print(f"[squat-collector] Saving CSV rows to: {dataset.resolve()}")
    print(f"[squat-collector] Existing rows: GOOD={good_count}, BAD={bad_count}")
    source = _capture_source(str(args.source))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    tracker = SquatRepTracker()
    controls = MouseControls()
    pending = None
    selected_errors: set[str] = set()
    status = f"Dataset ready: {dataset}"
    camera_text = "Move into a full-body side view."
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, controls.callback)

    with mp_pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok or frame is None:
                if isinstance(source, str):
                    break
                continue
            now = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 if isinstance(source, str) else time.monotonic()
            if not now:
                now = time.monotonic()
            if not isinstance(source, str):
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            assessment = assess_camera(
                results.pose_landmarks.landmark if results.pose_landmarks else None,
                now,
                preferred_side=tracker.locked_side,
                enforce_body_size=not tracker.active,
            )
            camera_text = assessment.guidance
            if pending is None:
                update = tracker.update(assessment.measurement if assessment.reliable else None)
                event = update["event"]
                if event and event["type"] == "calibrated":
                    status = "Standing calibrated. Begin when ready."
                elif event and event["type"] == "rejected":
                    status = f"Not saved: {event['reason']}"
                elif event and event["type"] == "completed":
                    features = build_rep_features(event["frames"], event["baseline"], event["total_frames"])
                    pending = {
                        "event": event,
                        "features": features,
                        "label_stage": "quality",
                        "collect_reasons": bool(args.collect_reasons),
                    }
                    selected_errors.clear()
                    status = "Rep complete; label it manually."
                    if args.debug:
                        print({"event": "rep_completed", "features": features})

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            display_frame = _draw_overlay(
                frame, tracker, camera_text, pending, selected_errors,
                good_count, bad_count, status, assessment.measurement, controls,
                dataset.name,
            )
            cv2.imshow(WINDOW_NAME, display_frame)
            quit_requested = False
            if pending is not None:
                # Freeze on the completed rep. This is especially important for
                # video input: choosing labels must not consume later frames.
                while pending is not None:
                    key = cv2.waitKey(30) & 0xFF
                    click_action = controls.consume()
                    if click_action == "good":
                        key = ord("g")
                    elif click_action == "bad_menu":
                        key = ord("b")
                    elif click_action == "discard":
                        key = ord("d")
                    elif click_action == "back":
                        pending["label_stage"] = "quality"
                    elif click_action == "bad_unsure":
                        selected_errors.clear()
                        key = ord("n")
                    elif click_action and click_action.startswith("reason:"):
                        selected_errors = {click_action.split(":", 1)[1]}
                        key = ord("n")
                    if key in (ord("q"), 27):
                        quit_requested = True
                        break
                    if key == ord("u"):
                        if undo_last_sample(dataset):
                            good_count, bad_count = sample_counts(dataset)
                            status = "Removed the most recently saved sample."
                        else:
                            status = "There is no saved sample to undo."
                    elif key == ord("b") and pending.get("label_stage") == "quality":
                        if pending.get("collect_reasons", False):
                            pending["label_stage"] = "reason"
                        else:
                            selected_errors.clear()
                            key = ord("n")
                    elif key in ERROR_KEYS and pending.get("label_stage") == "reason":
                        selected_errors = {ERROR_KEYS[key]}
                        key = ord("n")
                    if key in (ord("g"), ord("n")):
                        is_good = int(key == ord("g"))
                        errors = set() if is_good else selected_errors
                        event = pending["event"]
                        metadata = {
                            "participant_id": args.participant_id,
                            "session_id": args.session_id,
                            "recording_id": args.recording_id or Path(str(args.source)).stem,
                            "rep_number": event["rep_number"],
                            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                            "source": str(args.source),
                            "tracked_side": event["side"],
                        }
                        append_sample(dataset, metadata, pending["features"], is_good, errors)
                        good_count, bad_count = sample_counts(dataset)
                        label = "GOOD" if is_good else "BAD"
                        status = f"SAVED {label}. CSV now has {good_count + bad_count} rows."
                        print(
                            f"[squat-collector] SAVED {label} -> {dataset.resolve()} "
                            f"(GOOD={good_count}, BAD={bad_count})"
                        )
                        pending = None
                        selected_errors.clear()
                    elif key == ord("d"):
                        status = "Discarded completed rep without saving."
                        pending = None
                        selected_errors.clear()
                    if pending is not None:
                        frozen = _draw_overlay(
                            frame, tracker, camera_text, pending, selected_errors,
                            good_count, bad_count, status, assessment.measurement, controls,
                            dataset.name,
                        )
                        cv2.imshow(WINDOW_NAME, frozen)
            else:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    quit_requested = True
                elif key == ord("u"):
                    if undo_last_sample(dataset):
                        good_count, bad_count = sample_counts(dataset)
                        status = "Removed the most recently saved sample."
                    else:
                        status = "There is no saved sample to undo."
            if quit_requested:
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
