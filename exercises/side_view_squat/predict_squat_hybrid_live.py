"""Live side-view squat tracking with Random Forest quality plus rule feedback."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import mediapipe as mp

try:
    from .config import FEATURE_COLUMNS, MODEL_FILENAME, MODEL_VERSION, SQUAT_THRESHOLDS
    from .squat_camera import assess_camera
    from .squat_features import build_rep_features
    from .squat_hybrid import assess_completed_rep
    from .squat_rep_tracker import SquatRepTracker
except ImportError:
    from config import FEATURE_COLUMNS, MODEL_FILENAME, MODEL_VERSION, SQUAT_THRESHOLDS
    from squat_camera import assess_camera
    from squat_features import build_rep_features
    from squat_hybrid import assess_completed_rep
    from squat_rep_tracker import SquatRepTracker


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models" / MODEL_FILENAME
WINDOW_NAME = "VITAL-PT Side Squat Hybrid Coach"
DEFAULT_WINDOW_SIZE = (1280, 720)


class DebugLogger:
    def __init__(self, path: Path | None):
        self.handle = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = path.open("a", encoding="utf-8")

    def write(self, event: dict) -> None:
        if self.handle:
            self.handle.write(json.dumps(event, default=float) + "\n")
            self.handle.flush()

    def close(self) -> None:
        if self.handle:
            self.handle.close()


class FeedbackSpeaker:
    def __init__(self, enabled: bool, cooldown: float):
        self.enabled = enabled and os.name == "nt"
        self.cooldown = cooldown
        self.last_spoken = defaultdict(lambda: -float("inf"))

    def speak(self, text: str) -> None:
        now = time.monotonic()
        if not self.enabled or now - self.last_spoken[text] < self.cooldown:
            return
        self.last_spoken[text] = now
        escaped = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$speaker.Speak('{escaped}')"
        )
        subprocess.Popen(
            ["powershell", "-NoProfile", "-Command", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )


def load_bundle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}. Collect data and run train_squat_quality_model.py first.")
    bundle = joblib.load(path)
    if bundle.get("version") != MODEL_VERSION:
        print(f"WARNING: expected model version {MODEL_VERSION}, got {bundle.get('version')}")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Model feature order does not match the shared squat feature contract.")
    model = bundle.get("quality_model")
    if model is None:
        raise ValueError("Model bundle does not contain quality_model.")
    return bundle, model


def _put(image, text: str, y: int, color=(255, 255, 255), scale=0.55):
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_overlay(image, tracker, assessment, result):
    panel_height = 92 if result else 62
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (image.shape[1], panel_height), (25, 25, 25), -1)
    cv2.addWeighted(panel, 0.72, image, 0.28, 0, image)

    _put(image, f"{tracker.phase}  |  Reps: {tracker.rep_number}", 24, scale=0.5)
    if result:
        probability = "--" if result["good_probability"] is None else f"{result['good_probability']:.2f}"
        color = (130, 255, 130) if result["quality"] == "GOOD" else (120, 210, 255)
        if assessment.reliable:
            _put(image, f"Latest: {result['quality']}  |  P(good): {probability}", 52, color, 0.48)
            _put(image, result["feedback"], 80, color, 0.48)
        else:
            _put(image, assessment.guidance, 52, (120, 230, 255), 0.48)
            _put(image, f"Latest: {result['quality']}  |  P(good): {probability}", 80, color, 0.48)
    else:
        _put(image, assessment.guidance, 52, (120, 230, 255), 0.48)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=Path(os.getenv("SQUAT_MODEL", DEFAULT_MODEL)))
    parser.add_argument("--debug-log", type=Path, help="Append frame and rep diagnostics as JSON Lines.")
    parser.add_argument("--no-voice", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle, model = load_bundle(args.model)
    tracker = SquatRepTracker(bundle.get("hardcoded_rule_thresholds", {}).get("values"))
    logger = DebugLogger(args.debug_log)
    speaker = FeedbackSpeaker(not args.no_voice, SQUAT_THRESHOLDS["feedback_cooldown_seconds"])
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}.")
    latest_result = None
    previous_phase = tracker.phase
    fullscreen = False
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, *DEFAULT_WINDOW_SIZE)
        with mp_pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                now = time.monotonic()
                frame = cv2.flip(frame, 1)
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                assessment = assess_camera(
                    results.pose_landmarks.landmark if results.pose_landmarks else None,
                    now,
                    preferred_side=tracker.locked_side,
                )
                measurement = assessment.measurement
                update = tracker.update(measurement if assessment.reliable else None)
                logger.write({
                    "type": "frame",
                    "timestamp": now,
                    "camera_reliable": assessment.reliable,
                    "camera_guidance": assessment.guidance,
                    "phase": update["phase"],
                    "measurement": measurement,
                })
                if update["phase"] != previous_phase:
                    logger.write({
                        "type": "phase_change",
                        "timestamp": now,
                        "from": previous_phase,
                        "to": update["phase"],
                    })
                    previous_phase = update["phase"]
                event = update["event"]
                if event and event["type"] == "completed":
                    features = build_rep_features(event["frames"], event["baseline"], event["total_frames"])
                    latest_result = assess_completed_rep(
                        model, features, event["frames"], event["baseline"], event["rep_number"],
                        bundle.get("hardcoded_rule_thresholds", {}).get("values"),
                    )
                    logger.write({"type": "rep_result", "features": features, **latest_result})
                    print(json.dumps(latest_result, indent=2))
                    speaker.speak(latest_result["feedback"])
                elif event and event["type"] == "rejected":
                    logger.write({"type": "rep_rejected", **event})

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                _draw_overlay(frame, tracker, assessment, latest_result)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("f"):
                    fullscreen = not fullscreen
                    cv2.setWindowProperty(
                        WINDOW_NAME,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                    )
                elif key in (ord("q"), 27):
                    break
    finally:
        logger.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
