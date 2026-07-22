from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import joblib
from sklearn.ensemble import RandomForestClassifier

from exercises.side_view_squat.config import CSV_COLUMNS, ERROR_COLUMNS, FEATURE_COLUMNS
from exercises.side_view_squat.squat_camera import LANDMARK_INDEX, assess_camera, choose_visible_side
from exercises.side_view_squat.squat_collection import append_sample, ensure_dataset
from exercises.side_view_squat.squat_features import build_rep_features, ordered_feature_vector
from exercises.side_view_squat.squat_feedback_rules import evaluate_rules, primary_rule
from exercises.side_view_squat.squat_geometry import calculate_angle
from exercises.side_view_squat.squat_hybrid import assess_completed_rep
from exercises.side_view_squat.squat_rep_tracker import SquatRepTracker
from exercises.side_view_squat.export_squat_onnx import export


def frame(timestamp, knee, hip_y, phase="DESCENDING", hip=160.0, torso=12.0, visibility=0.95):
    return {
        "timestamp": float(timestamp),
        "knee_angle": float(knee),
        "hip_angle": float(hip),
        "ankle_angle": 105.0,
        "torso_lean": float(torso),
        "shoulder_y": hip_y - 0.30,
        "hip_y": float(hip_y),
        "knee_y": 0.70,
        "ankle_y": 0.90,
        "heel_y": 0.90,
        "foot_y": 0.91,
        "body_scale": 0.50,
        "visibility": float(visibility),
        "valid": True,
        "side": "left",
        "phase": phase,
    }


BASELINE = {
    "knee_angle": 170.0,
    "hip_angle": 170.0,
    "hip_y": 0.40,
    "torso_lean": 8.0,
    "body_scale": 0.50,
    "heel_foot_offset": 0.01,
}


def completed_frames():
    values = [
        (0.0, 150, 0.44, "DESCENDING"),
        (0.1, 130, 0.49, "DESCENDING"),
        (0.2, 105, 0.55, "DESCENDING"),
        (0.3, 92, 0.60, "BOTTOM"),
        (0.4, 92, 0.60, "BOTTOM"),
        (0.5, 110, 0.56, "ASCENDING"),
        (0.6, 135, 0.49, "ASCENDING"),
        (0.8, 158, 0.42, "ASCENDING"),
        (1.0, 168, 0.40, "ASCENDING"),
    ]
    return [frame(*item) for item in values]


def good_features():
    return build_rep_features(completed_frames(), BASELINE)


class ConstantModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, values):
        return np.asarray([[1.0 - self.probability, self.probability] for _ in values])


def test_angle_calculation():
    assert calculate_angle((1, 0), (0, 0), (0, 1)) == pytest.approx(90.0)
    assert calculate_angle((-1, 0), (0, 0), (1, 0)) == pytest.approx(180.0)


def test_side_selection_prefers_visible_side_and_holds_with_margin():
    named = {}
    for side, score in (("left", 0.90), ("right", 0.60)):
        for part in ("shoulder", "hip", "knee", "ankle", "heel", "foot"):
            named[f"{side}_{part}"] = SimpleNamespace(x=0.5, y=0.5, visibility=score)
    assert choose_visible_side(named) == "left"
    assert choose_visible_side(named, preferred_side="right", switch_margin=0.35) == "right"


def test_camera_rejects_low_visibility():
    landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.95) for _ in range(33)]
    for name, index in LANDMARK_INDEX.items():
        point = landmarks[index]
        point.x = 0.48 if name.startswith("left") else 0.50
        point.y = {"shoulder": 0.15, "hip": 0.42, "knee": 0.65, "ankle": 0.87, "heel": 0.89, "foot": 0.88}[name.split("_")[-1]]
    landmarks[LANDMARK_INDEX["left_foot"]].visibility = 0.1
    landmarks[LANDMARK_INDEX["right_foot"]].visibility = 0.1
    assessment = assess_camera(landmarks, 0.0)
    assert not assessment.reliable
    assert "lighting" in assessment.guidance.lower()


def _tracker():
    return SquatRepTracker({
        "calibration_frames": 5,
        "phase_confirm_frames": 2,
        "return_confirm_frames": 2,
        "minimum_valid_frames": 8,
        "minimum_rep_duration": 0.5,
    })


def _tracker_frame(timestamp, knee, hip_y, hip=170.0):
    return frame(timestamp, knee, hip_y, hip=hip, phase="STANDING")


def test_rep_state_transitions_and_completion():
    tracker = _tracker()
    for index in range(5):
        tracker.update(_tracker_frame(index * 0.1, 170, 0.40))
    assert tracker.phase == "STANDING"
    sequence = [
        (0.5, 157, 0.43), (0.6, 145, 0.46),
        (0.7, 122, 0.53), (0.8, 95, 0.59), (0.9, 95, 0.59), (1.0, 95, 0.59),
        (1.1, 112, 0.55), (1.2, 135, 0.49),
        (1.3, 158, 0.42), (1.4, 166, 0.40),
        (1.5, 169, 0.40), (1.6, 170, 0.40), (1.7, 170, 0.40),
    ]
    event = None
    phases = []
    for values in sequence:
        update = tracker.update(_tracker_frame(*values))
        phases.append(update["phase"])
        event = update["event"] or event
    assert "DESCENDING" in phases
    assert "BOTTOM" in phases
    assert "ASCENDING" in phases
    assert event and event["type"] == "completed"


def test_tiny_knee_bend_does_not_count_and_hysteresis_blocks_one_frame_start():
    tracker = _tracker()
    for index in range(5):
        tracker.update(_tracker_frame(index * 0.1, 170, 0.40))
    tracker.update(_tracker_frame(0.5, 154, 0.43))
    assert tracker.phase == "STANDING"
    tracker.update(_tracker_frame(0.6, 169, 0.40))
    assert tracker.rep_number == 0


def test_feature_order_and_values_are_stable():
    features = good_features()
    assert list(features) == FEATURE_COLUMNS
    assert len(ordered_feature_vector(features)) == len(FEATURE_COLUMNS)
    assert features["knee_rom"] > 50
    assert features["valid_frame_count"] == len(completed_frames())


def test_csv_feature_order(tmp_path: Path):
    path = ensure_dataset(tmp_path / "samples.csv")
    metadata = {
        "participant_id": "p1", "session_id": "s1", "recording_id": "r1",
        "rep_number": 1, "recorded_at_utc": "2026-01-01T00:00:00Z",
        "source": "camera", "tracked_side": "left",
    }
    append_sample(path, metadata, good_features(), 1)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == CSV_COLUMNS
        assert next(reader)["is_good"] == "1"


def test_model_input_shape_matches_contract():
    X = np.random.default_rng(42).normal(size=(24, len(FEATURE_COLUMNS)))
    y = np.asarray([0, 1] * 12)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    assert model.n_features_in_ == len(FEATURE_COLUMNS)


def test_shallow_depth_requires_multiple_signals():
    features = good_features()
    features.update(knee_angle_at_bottom=140.0, knee_rom=35.0, hip_knee_depth_at_bottom=-0.2)
    rules = evaluate_rules(features)
    assert primary_rule(rules).name == "shallow_depth"


def test_heel_lift_detection_requires_persistence():
    features = good_features()
    features.update(heel_lift_max=0.08, heel_lift_bottom_fraction=0.75)
    assert any(rule.name == "heel_lift" for rule in evaluate_rules(features))
    features["heel_lift_bottom_fraction"] = 0.1
    assert not any(rule.name == "heel_lift" for rule in evaluate_rules(features))


def test_torso_lean_detection_is_relative_and_lenient():
    features = good_features()
    features.update(standing_torso_lean=8.0, max_torso_lean=65.0)
    assert any(rule.name == "excessive_torso_lean" for rule in evaluate_rules(features))
    features.update(standing_torso_lean=35.0, max_torso_lean=60.0)
    assert not any(rule.name == "excessive_torso_lean" for rule in evaluate_rules(features))


def test_chest_collapse_requires_rise_and_lean_evidence():
    features = good_features()
    features["hip_shoulder_rise_difference"] = 0.16
    frames = completed_frames()
    for index, item in enumerate([entry for entry in frames if entry["phase"] == "ASCENDING"]):
        item["torso_lean"] = 12.0 + index * 6.0
    assert any(rule.name == "chest_collapse" for rule in evaluate_rules(features, frames, BASELINE))


def test_fast_rep_needs_two_signals():
    features = good_features()
    features.update(rep_duration=0.7, descent_duration=0.2, max_descent_knee_velocity=250.0)
    assert any(rule.name == "uncontrolled_tempo" for rule in evaluate_rules(features))
    features.update(rep_duration=1.2, descent_duration=0.5)
    assert not any(rule.name == "uncontrolled_tempo" for rule in evaluate_rules(features))


def test_unreliable_rep_is_unscored_before_model():
    features = good_features()
    features["pose_visibility_mean"] = 0.4
    result = assess_completed_rep(ConstantModel(0.1), features)
    assert result["quality"] == "UNSCORED"
    assert result["good_probability"] is None


def test_borderline_probability_is_not_forced_bad_by_weak_or_no_rule():
    result = assess_completed_rep(
        ConstantModel(0.5), good_features(), thresholds={"minimum_valid_frames": 8}
    )
    assert result["quality"] == "BORDERLINE"
    assert result["primary_error"] is None


def test_only_one_primary_feedback_is_selected():
    features = good_features()
    features.update(
        knee_angle_at_bottom=145.0, knee_rom=30.0, hip_knee_depth_at_bottom=-0.2,
        heel_lift_max=0.10, heel_lift_bottom_fraction=1.0,
    )
    result = assess_completed_rep(
        ConstantModel(0.1), features, thresholds={"minimum_valid_frames": 8}
    )
    assert result["quality"] == "BAD"
    assert result["primary_error"] in {"shallow_depth", "heel_lift"}
    assert isinstance(result["feedback"], str) and "\n" not in result["feedback"]
    assert len(result["triggered_rules"]) >= 2


def test_python_to_browser_onnx_manifest_preserves_feature_order(tmp_path: Path):
    X = np.random.default_rng(7).normal(size=(24, len(FEATURE_COLUMNS)))
    y = np.asarray([0, 1] * 12)
    model = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)
    model_path = tmp_path / "bundle.pkl"
    joblib.dump({
        "quality_model": model,
        "feature_columns": FEATURE_COLUMNS,
        "version": "side_squat_hybrid_quality_v1",
        "probability_thresholds": {"high_confidence_good": 0.65, "high_confidence_bad": 0.35},
        "hardcoded_rule_thresholds": {"values": {}, "documentation": {}},
    }, model_path)
    onnx_path, manifest_path = export(model_path, tmp_path / "browser")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert onnx_path.exists()
    assert manifest["feature_names"] == FEATURE_COLUMNS
    assert manifest["input"]["shape"] == [None, len(FEATURE_COLUMNS)]
    assert manifest["class_order"] == [0, 1]
