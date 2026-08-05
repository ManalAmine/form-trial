"""Interactive webcam/video collection of one CSV row per completed push-up rep."""

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
    from .config import COLLECTOR_VERSION
    from .push_up_camera import assess_camera
    from .push_up_collection import append_sample, ensure_dataset, sample_counts, undo_last_sample
    from .push_up_features import build_rep_features
    from .push_up_state_machine import PushUpStateMachine
except ImportError:
    from config import COLLECTOR_VERSION
    from push_up_camera import assess_camera
    from push_up_collection import append_sample, ensure_dataset, sample_counts, undo_last_sample
    from push_up_features import build_rep_features
    from push_up_state_machine import PushUpStateMachine


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "data" / "push_up_reps.csv"
WINDOW_NAME = "VITAL-PT Side Push-Up Data Collection"


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
    parser.add_argument("--camera", type=int, help="Compatibility alias for a webcam index; overrides --source.")
    parser.add_argument("--dataset", default=os.getenv("PUSH_UP_DATASET", str(DEFAULT_DATASET)))
    parser.add_argument("--subject-id", default=os.getenv("SUBJECT_ID", "p001"))
    parser.add_argument("--session-id", default=os.getenv("SESSION_ID", datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--debug", action="store_true", help="Print completed-rep features to the terminal.")
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
    measurement=None, controls=None, dataset_name="push_up_reps.csv",
):
    """Render the same fixed 1200x720 dashboard used by the squat collector."""
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
        "WAITING_FOR_POSITION": ("GET READY", "Hold a high-plank position", (40, 150, 220)),
        "TOP": ("READY", "Start your push-up", (40, 170, 70)),
        "DESCENDING": ("REP IN PROGRESS", "Lower with control", (50, 150, 220)),
        "BOTTOM": ("BOTTOM FOUND", "Now press back up", (50, 150, 220)),
        "ASCENDING": ("REP IN PROGRESS", "Finish with straight arms", (50, 150, 220)),
    }
    banner, instruction, banner_color = phase_messages.get(
        tracker.phase, (tracker.phase, "", (70, 70, 70))
    )
    if pending:
        banner, instruction, banner_color = (
            "REP CAPTURED!", "Click GOOD or BAD to save this row", (35, 150, 55)
        )
    cv2.rectangle(canvas, (0, 0), (1200, 70), banner_color, -1)
    _put(canvas, banner, 34, (255, 255, 255), 0.92, 20, 330, 2)
    _put(canvas, instruction, 35, (255, 255, 255), 0.72, 390, 770, 2)

    panel_x, panel_width = 18, 330
    _put(canvas, f"Phase: {tracker.phase}", 108, (255, 235, 120), 0.66, panel_x, panel_width)
    if measurement:
        _put(canvas, f"Tracked side: {measurement['side']}", 140, (255, 255, 255), 0.58, panel_x, panel_width)
        _put(canvas, f"Facing: {measurement['facing_direction']}", 169, (255, 255, 255), 0.58, panel_x, panel_width)
        _put(canvas, f"Tracking quality: {measurement['visibility']:.0%}", 198, (210, 210, 210), 0.50, panel_x, panel_width)
    else:
        _put(canvas, "Pose tracking: not available", 140, (180, 180, 180), 0.52, panel_x, panel_width)

    if pending:
        _put(canvas, "HOW WAS THIS REP?", 270, (255, 255, 255), 0.68, panel_x, panel_width)
        good_button = (18, 290, 348, 370)
        bad_button = (18, 385, 348, 465)
        _draw_button(canvas, good_button, "GOOD FORM", (35, 150, 55))
        _draw_button(canvas, bad_button, "BAD FORM", (45, 45, 190))
        _put(canvas, "Clicking saves one CSV row now.", 505, (180, 255, 180), 0.48, panel_x, panel_width)
        _put(canvas, "D discards without saving.", 535, (220, 220, 220), 0.43, panel_x, panel_width)
        if controls:
            controls.set_buttons({
                "good": good_button,
                "bad": bad_button,
            })
    else:
        if controls:
            controls.set_buttons({})
        _put(canvas, "GOOD / BAD SAVE", 270, (255, 255, 255), 0.68, panel_x, panel_width)
        good_button = (18, 290, 348, 370)
        bad_button = (18, 385, 348, 465)
        _draw_button(canvas, good_button, "GOOD FORM", (70, 95, 75))
        _draw_button(canvas, bad_button, "BAD FORM", (85, 70, 95))
        _put(canvas, "Complete a rep to enable buttons.", 500, (120, 230, 255), 0.46, panel_x, panel_width)
        _put(canvas, "Hold plank > lower > press up.", 532, (220, 220, 220), 0.43, panel_x, panel_width)
        _put(canvas, "R resets the current attempt.", 562, (190, 190, 190), 0.43, panel_x, panel_width)

    _put(canvas, f"Detected reps: {tracker.rep_number}", 610, (255, 255, 255), 0.54, panel_x, panel_width)
    _put(canvas, f"CSV rows: GOOD {good}  BAD {bad}", 640, (255, 255, 255), 0.54, panel_x, panel_width)
    _put(canvas, f"CSV file: {dataset_name}", 668, (190, 190, 190), 0.43, panel_x, panel_width)
    _put(canvas, "Q quit | U undo last save", 698, (190, 190, 190), 0.43, panel_x, panel_width)

    guidance_color = (120, 230, 255) if camera_text == "Camera position is good" else (90, 180, 255)
    cv2.rectangle(canvas, (370, 675), (1180, 714), (35, 35, 38), -1)
    _put(canvas, f"Camera: {camera_text}", 704, guidance_color, 0.56, 382, 780)
    if status:
        _put(canvas, f"Status: {status}", 65, (240, 240, 240), 0.42, 20, 330, 1)
    return canvas


def run(args) -> None:
    dataset = ensure_dataset(args.dataset)
    good_count, bad_count, _ = sample_counts(dataset)
    source_value = str(args.camera) if args.camera is not None else str(args.source)
    source = _capture_source(source_value)
    print(f"[push-up-collector] Saving CSV rows to: {dataset.resolve()}")
    print(f"[push-up-collector] Existing rows: GOOD={good_count}, BAD={bad_count}")
    print("[push-up-collector] Controls: G good | B bad | D discard | U undo | R reset | Q quit")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source_value}")

    tracker = PushUpStateMachine()
    controls = MouseControls()
    pending = None
    status = f"Dataset ready: {dataset}"
    camera_text = "Move into a full-body side view."
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, controls.callback)

    try:
        with mp_pose.Pose(
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
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
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                assessment = assess_camera(
                    results.pose_landmarks.landmark if results.pose_landmarks else None,
                    now,
                    preferred_side=tracker.locked_side,
                    lock_side=tracker.active,
                )
                camera_text = assessment.guidance
                if pending is None:
                    update = tracker.update(assessment.measurement if assessment.reliable else None)
                    event = update["event"]
                    if event and event["type"] == "rejected":
                        status = f"Not saved: {event['reason']}"
                    elif event and event["type"] == "completed":
                        features = build_rep_features(event["frames"], event["total_frames"])
                        pending = {
                            "event": event,
                            "features": features,
                        }
                        status = "Rep complete; label it manually."
                        if args.debug:
                            print({"event": "rep_completed", "features": features})

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                display_frame = _draw_overlay(
                    frame, tracker, camera_text, pending, set(),
                    good_count, bad_count, status, assessment.measurement, controls,
                    dataset.name,
                )
                cv2.imshow(WINDOW_NAME, display_frame)
                quit_requested = False

                if pending is not None:
                    # Freeze on the completed rep so video input cannot advance
                    # while the developer chooses GOOD, BAD, or DISCARD.
                    while pending is not None:
                        key = cv2.waitKey(30) & 0xFF
                        click_action = controls.consume()
                        if click_action == "good":
                            key = ord("g")
                        elif click_action == "bad":
                            key = ord("b")

                        if key in (ord("q"), 27):
                            quit_requested = True
                            break
                        if key == ord("u"):
                            if undo_last_sample(dataset):
                                good_count, bad_count, _ = sample_counts(dataset)
                                status = "Removed the most recently saved sample."
                            else:
                                status = "There is no saved sample to undo."
                        should_save_good = key == ord("g")
                        should_save_bad = key == ord("b")
                        if should_save_good or should_save_bad:
                            is_good = int(should_save_good)
                            event = pending["event"]
                            metadata = {
                                "subject_id": args.subject_id,
                                "session_id": args.session_id,
                                "rep_index": good_count + bad_count + 1,
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "selected_side": event["side"],
                                "facing_direction": event["facing_direction"],
                                "collector_version": COLLECTOR_VERSION,
                            }
                            append_sample(dataset, metadata, pending["features"], is_good, set())
                            good_count, bad_count, _ = sample_counts(dataset)
                            label = "GOOD" if is_good else "BAD"
                            status = f"SAVED {label}. CSV now has {good_count + bad_count} rows."
                            print(
                                f"[push-up-collector] SAVED {label} -> {dataset.resolve()} "
                                f"(GOOD={good_count}, BAD={bad_count})"
                            )
                            pending = None
                        elif key == ord("d"):
                            status = "Discarded completed rep without saving."
                            pending = None

                        if pending is not None:
                            frozen = _draw_overlay(
                                frame, tracker, camera_text, pending, set(),
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
                            good_count, bad_count, _ = sample_counts(dataset)
                            status = "Removed the most recently saved sample."
                        else:
                            status = "There is no saved sample to undo."
                    elif key == ord("r"):
                        tracker.reset_attempt()
                        status = "Current attempt reset. Hold a high plank."
                if quit_requested:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
