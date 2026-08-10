"""Live Python side-view push-up tracking with hybrid quality feedback."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

try:
    from .config import BUNDLE_VERSION, FEATURE_COLUMNS, MODEL_FILENAME, PUSH_UP_THRESHOLDS
    from .push_up_camera import assess_camera
    from .push_up_features import build_rep_features
    from .push_up_hybrid import assess_completed_rep
    from .push_up_state_machine import PushUpStateMachine
except ImportError:
    from config import BUNDLE_VERSION, FEATURE_COLUMNS, MODEL_FILENAME, PUSH_UP_THRESHOLDS
    from push_up_camera import assess_camera
    from push_up_features import build_rep_features
    from push_up_hybrid import assess_completed_rep
    from push_up_state_machine import PushUpStateMachine


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models" / MODEL_FILENAME
WINDOW_NAME = "VITAL-PT Side-View Push-Up Hybrid Coach"


class FeedbackSpeaker:
    """Non-blocking Windows speech worker, matching the squat live coach."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and os.name == "nt"
        self.process = None
        if self.enabled:
            worker_script = (
                "$speaker = New-Object -ComObject SAPI.SpVoice; "
                "while (($line = [Console]::In.ReadLine()) -ne $null) { "
                "if ($line -eq '__VITAL_PT_QUIT__') { break }; "
                "[void]$speaker.Speak($line) "
                "}"
            )
            try:
                self.process = subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", worker_script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    encoding="utf-8",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError:
                self.enabled = False

    def speak(self, text: str) -> bool:
        if not self.enabled or not self.process or self.process.poll() is not None or not self.process.stdin:
            return False
        try:
            self.process.stdin.write(text.replace("\r", " ").replace("\n", " ") + "\n")
            self.process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            self.enabled = False
            return False

    def close(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            self.process.terminate()
        self.process = None


def voice_feedback_text(result: dict) -> str:
    if result["final_quality"] == "GOOD":
        return "Good rep."
    if result["final_quality"] == "BAD":
        return f"Bad rep. {result['feedback']}."
    return "Rep not scored. Make sure your full body is visible."


def load_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Push-up model bundle not found: {path}. Collect labelled data and run train_push_up_model.py first."
        )
    bundle = joblib.load(path)
    version = bundle.get("bundle_version", bundle.get("version"))
    if version != BUNDLE_VERSION:
        raise ValueError(f"Incompatible push-up bundle version: expected {BUNDLE_VERSION}, got {version}.")
    if bundle.get("exercise_name") != "push_up":
        raise ValueError("Model bundle is not a push-up model.")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature order does not exactly match the push-up feature contract.")
    model = bundle.get("quality_model")
    if model is None:
        raise ValueError("Model bundle does not contain quality_model.")
    if getattr(model, "n_features_in_", len(FEATURE_COLUMNS)) != len(FEATURE_COLUMNS):
        raise ValueError("Model input width is incompatible with the push-up feature contract.")
    thresholds = bundle.get("hardcoded_thresholds", {}).get("values")
    if not isinstance(thresholds, dict):
        raise ValueError("Model bundle is missing hardcoded push-up thresholds.")
    return bundle, model, thresholds


def print_thresholds(thresholds: dict) -> None:
    print("Loaded push-up thresholds:")
    for name in sorted(thresholds):
        print(f"  {name}={thresholds[name]}")


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


def _draw_overlay(image, tracker, assessment, result, version, rejection=None):
    """Render the same wide, undistorted 1200x720 dashboard as collection."""
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
    cv2.rectangle(canvas, (0, 0), (1200, 70), banner_color, -1)
    _put(canvas, banner, 34, (255, 255, 255), 0.92, 20, 330, 2)
    _put(canvas, instruction, 35, (255, 255, 255), 0.72, 390, 770, 2)

    panel_x, panel_width = 18, 330
    _put(canvas, f"Phase: {tracker.phase}", 108, (255, 235, 120), 0.66, panel_x, panel_width)
    if assessment.measurement:
        _put(canvas, f"Tracked side: {assessment.side}", 140, (255, 255, 255), 0.58, panel_x, panel_width)
        _put(canvas, f"Facing: {assessment.facing_direction}", 169, (255, 255, 255), 0.58, panel_x, panel_width)
        _put(canvas, "Pose tracking: ready", 198, (210, 210, 210), 0.50, panel_x, panel_width)
    else:
        _put(canvas, "Pose tracking: not available", 140, (180, 180, 180), 0.52, panel_x, panel_width)

    _put(canvas, "LATEST RESULT", 255, (255, 255, 255), 0.64, panel_x, panel_width)
    if result:
        is_good = result["final_quality"] == "GOOD"
        result_color = (35, 150, 55) if is_good else (45, 45, 190)
        cv2.rectangle(canvas, (18, 275), (348, 355), result_color, -1)
        cv2.rectangle(canvas, (18, 275), (348, 355), (255, 255, 255), 2)
        _put(canvas, result["final_quality"], 329, (255, 255, 255), 1.25, 72, 230, 3)
        _put(canvas, "WHY", 405, (210, 210, 210), 0.52, panel_x, panel_width)
        _put(canvas, result["feedback"], 455, (120, 255, 150) if is_good else (120, 200, 255), 0.72, panel_x, panel_width)
        if not is_good and result["strongest_error"] is None:
            _put(canvas, "No specific rule was confident.", 495, (190, 190, 190), 0.43, panel_x, panel_width)
    else:
        cv2.rectangle(canvas, (18, 275), (348, 355), (75, 75, 78), -1)
        cv2.rectangle(canvas, (18, 275), (348, 355), (150, 150, 150), 2)
        _put(canvas, "WAITING FOR REP", 326, (220, 220, 220), 0.72, 42, 280, 2)
        _put(canvas, "Complete one full down-and-up cycle.", 393, (210, 210, 210), 0.47, panel_x, panel_width)
    if rejection:
        _put(canvas, f"Not scored: {rejection}", 585, (100, 180, 255), 0.43, panel_x, panel_width)

    _put(canvas, f"Repetitions: {tracker.rep_number}", 620, (255, 255, 255), 0.54, panel_x, panel_width)
    _put(canvas, f"Bundle: {version}", 655, (190, 190, 190), 0.40, panel_x, panel_width)
    _put(canvas, "Q or Esc: quit", 690, (190, 190, 190), 0.43, panel_x, panel_width)

    guidance_color = (120, 230, 255) if assessment.guidance == "Camera position is good" else (90, 180, 255)
    cv2.rectangle(canvas, (370, 675), (1180, 714), (35, 35, 38), -1)
    _put(canvas, f"Camera: {assessment.guidance}", 704, guidance_color, 0.56, 382, 780)
    return canvas


def print_rep_summary(result: dict) -> None:
    print(f"Rep {result['rep_number']}")
    print(f"Final result: {result['final_quality']}")
    print(f"Reason: {result['feedback']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path(os.getenv("PUSH_UP_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken rep feedback.")
    parser.add_argument("--debug", action="store_true", help="Print internal thresholds at startup.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle, model, thresholds = load_bundle(args.model)
    if args.debug:
        print_thresholds(thresholds)
    tracker = PushUpStateMachine(thresholds)
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")
    speaker = FeedbackSpeaker(enabled=not args.no_voice)
    if speaker.enabled:
        print("Voice feedback: enabled")
    else:
        print("Voice feedback: disabled")
    latest_result = None
    rejection = None
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        with mp_pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frame = cv2.flip(frame, 1)
                timestamp = cv2.getTickCount() / cv2.getTickFrequency()
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                assessment = assess_camera(
                    results.pose_landmarks.landmark if results.pose_landmarks else None,
                    timestamp,
                    preferred_side=tracker.locked_side,
                    thresholds=thresholds,
                    lock_side=tracker.active,
                )
                update = tracker.update(assessment.measurement if assessment.reliable else None)
                event = update["event"]
                if event and event["type"] == "completed":
                    features = build_rep_features(event["frames"], event["total_frames"], thresholds)
                    latest_result = assess_completed_rep(model, features, event["rep_number"], thresholds)
                    print_rep_summary(latest_result)
                    speaker.speak(voice_feedback_text(latest_result))
                    rejection = None
                elif event and event["type"] == "rejected":
                    rejection = event["reason"]
                    print(f"Attempt rejected: {rejection}")
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                display_frame = _draw_overlay(
                    frame, tracker, assessment, latest_result,
                    bundle["bundle_version"], rejection,
                )
                cv2.imshow(WINDOW_NAME, display_frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        speaker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
