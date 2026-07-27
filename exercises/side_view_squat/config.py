"""Single source of truth for the side-view squat feature and threshold contract."""

from __future__ import annotations

from copy import deepcopy

EXERCISE_ID = "side-view-squat"
EXERCISE_NAME = "side_view_squat"
MODEL_VERSION = "side_squat_hybrid_quality_v1"
MODEL_FILENAME = "side_squat_hybrid_quality_v1.pkl"

METADATA_COLUMNS = [
    "participant_id",
    "session_id",
    "recording_id",
    "rep_number",
    "recorded_at_utc",
    "source",
    "tracked_side",
]

# This exact order is used for CSV writing, sklearn inference, ONNX input, and the
# browser manifest. Changing it requires a new model version.
FEATURE_COLUMNS = [
    "min_knee_angle",
    "max_knee_angle",
    "knee_rom",
    "knee_angle_at_bottom",
    "max_descent_knee_velocity",
    "max_ascent_knee_velocity",
    "min_hip_angle",
    "max_hip_angle",
    "hip_rom",
    "hip_angle_at_bottom",
    "hip_vertical_drop",
    "hip_knee_depth_at_bottom",
    "min_ankle_angle",
    "max_ankle_angle",
    "ankle_rom",
    "ankle_angle_at_bottom",
    "standing_torso_lean",
    "torso_lean_at_bottom",
    "max_torso_lean",
    "torso_lean_range",
    "heel_lift_max",
    "heel_lift_bottom_fraction",
    "hip_shoulder_rise_difference",
    "movement_smoothness",
    "rep_duration",
    "descent_duration",
    "bottom_duration",
    "ascent_duration",
    "descent_ascent_ratio",
    "pose_visibility_mean",
    "pose_visibility_min",
    "valid_frame_ratio",
    "valid_frame_count",
]

ERROR_COLUMNS = [
    "err_shallow_depth",
    "err_incomplete_lockout",
    "err_heel_lift",
    "err_excessive_torso_lean",
    "err_chest_collapse",
    "err_uncontrolled_tempo",
]

TARGET_COLUMN = "is_good"
CSV_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + ERROR_COLUMNS + [TARGET_COLUMN]


# Values are deliberately lenient initial defaults. Distances are normalized by
# the calibrated shoulder-to-ankle body scale; angles are degrees; times seconds;
# velocity is degrees/second. `description` explains strictness direction.
THRESHOLD_SPECS = {
    "calibration_frames": {
        "value": 20,
        "unit": "frames",
        "description": "Stable standing frames required. Increasing improves calibration but delays readiness.",
    },
    "phase_confirm_frames": {
        "value": 3,
        "unit": "frames",
        "description": "Consecutive frames required for a phase change. Increasing is more tolerant of jitter.",
    },
    "bottom_confirm_frames": {
        "value": 2,
        "unit": "frames",
        "description": "Consecutive bottom frames required. Increasing rejects more bounce-style reversals.",
    },
    "ascent_confirm_frames": {
        "value": 2,
        "unit": "frames",
        "description": "Consecutive rising frames required. Increasing is more tolerant of velocity jitter.",
    },
    "measurement_smoothing_window": {
        "value": 3,
        "unit": "frames",
        "description": "Median window for state and feature measurements. Increasing smooths more but adds lag.",
    },
    "maximum_consecutive_invalid_frames": {
        "value": 8,
        "unit": "frames",
        "description": "Pose-loss frames allowed mid-rep. Increasing is more tolerant of brief occlusion.",
    },
    "return_confirm_frames": {
        "value": 3,
        "unit": "frames",
        "description": "Consecutive upright frames required to finish. Increasing rejects brief false lockouts.",
    },
    "visibility_threshold": {
        "value": 0.55,
        "unit": "MediaPipe confidence 0-1",
        "description": "Minimum mean visibility across the tracked side. Increasing is stricter about tracking.",
    },
    "core_visibility_threshold": {
        "value": 0.35,
        "unit": "MediaPipe confidence 0-1",
        "description": "Minimum visibility for shoulder, hip, knee, and ankle. Heel or foot confidence alone does not reject a rep.",
    },
    "reliable_visibility_mean": {
        "value": 0.72,
        "unit": "MediaPipe confidence 0-1",
        "description": "Minimum mean visibility for scoring. Increasing is stricter about pose quality.",
    },
    "reliable_visibility_min": {
        "value": 0.35,
        "unit": "MediaPipe confidence 0-1",
        "description": "Minimum observed visibility for scoring. Increasing is stricter.",
    },
    "minimum_valid_frame_ratio": {
        "value": 0.85,
        "unit": "fraction",
        "description": "Required fraction of valid frames. Increasing is stricter.",
    },
    "minimum_valid_frames": {
        "value": 15,
        "unit": "frames",
        "description": "Minimum samples in a completed rep. Increasing rejects more short reps.",
    },
    "maximum_side_width_ratio": {
        "value": 0.34,
        "unit": "shoulder/hip width divided by body scale",
        "description": "Maximum apparent left-right width for side-on placement. Increasing is more tolerant of camera angle.",
    },
    "side_switch_visibility_margin": {
        "value": 0.12,
        "unit": "MediaPipe confidence 0-1",
        "description": "Visibility advantage needed to switch sides while idle. Increasing favors side stability.",
    },
    "minimum_body_frame_fraction": {
        "value": 0.42,
        "unit": "fraction of frame height",
        "description": "Minimum body height in frame. Increasing rejects cameras that are farther away.",
    },
    "maximum_body_frame_fraction": {
        "value": 0.94,
        "unit": "fraction of frame height",
        "description": "Maximum body height in frame. Increasing tolerates a closer camera.",
    },
    "frame_edge_margin": {
        "value": 0.025,
        "unit": "normalized image coordinate",
        "description": "Required clearance from image edges. Increasing requires more space around the body.",
    },
    "standing_knee_min": {
        "value": 150.0,
        "unit": "degrees",
        "description": "Minimum knee angle accepted during calibration. Increasing demands a straighter stance.",
    },
    "standing_hip_min": {
        "value": 145.0,
        "unit": "degrees",
        "description": "Minimum hip angle accepted during calibration. Increasing demands a straighter stance.",
    },
    "descent_knee_delta": {
        "value": 10.0,
        "unit": "degrees below standing baseline",
        "description": "Bend needed to start descent. Increasing ignores more small knee bends.",
    },
    "descent_hip_drop": {
        "value": 0.025,
        "unit": "body scale",
        "description": "Hip drop supporting descent detection. Increasing ignores more small movements.",
    },
    "minimum_rep_rom": {
        "value": 42.0,
        "unit": "degrees",
        "description": "Minimum knee range for a countable rep. Increasing rejects more partial attempts.",
    },
    "bottom_knee_angle": {
        "value": 122.0,
        "unit": "degrees",
        "description": "Knee angle that can enter BOTTOM after adequate ROM. Increasing recognizes bottom earlier.",
    },
    "bottom_hip_drop": {
        "value": 0.10,
        "unit": "body scale",
        "description": "Hip displacement that supports bottom detection. Increasing requires more descent.",
    },
    "bottom_velocity": {
        "value": 45.0,
        "unit": "degrees/second",
        "description": "Maximum absolute knee velocity near bottom. Increasing is more tolerant of no pause.",
    },
    "ascent_velocity": {
        "value": 12.0,
        "unit": "degrees/second",
        "description": "Positive knee velocity needed for ascent. Increasing is stricter.",
    },
    "return_knee_tolerance": {
        "value": 14.0,
        "unit": "degrees below calibrated standing",
        "description": "Allowed final knee flexion. Increasing is more tolerant.",
    },
    "return_hip_tolerance": {
        "value": 16.0,
        "unit": "degrees below calibrated standing",
        "description": "Allowed final hip flexion. Increasing is more tolerant.",
    },
    "lockout_knee_delta": {
        "value": 10.0,
        "unit": "degrees below calibrated standing",
        "description": "Clearly flexed final knee used for feedback. Increasing is more tolerant.",
    },
    "lockout_hip_delta": {
        "value": 12.0,
        "unit": "degrees below calibrated standing",
        "description": "Clearly flexed final hip used for feedback. Increasing is more tolerant.",
    },
    "maximum_rep_duration": {
        "value": 10.0,
        "unit": "seconds",
        "description": "Timeout for an attempt. Increasing permits slower reps.",
    },
    "minimum_rep_duration": {
        "value": 0.65,
        "unit": "seconds",
        "description": "Shortest reliably scoreable rep. Decreasing permits faster reps.",
    },
    "shallow_knee_angle": {
        "value": 128.0,
        "unit": "degrees",
        "description": "One shallow-depth signal. Decreasing makes the rule more tolerant.",
    },
    "shallow_min_rom": {
        "value": 55.0,
        "unit": "degrees",
        "description": "Second shallow-depth signal. Decreasing makes the rule more tolerant.",
    },
    "shallow_hip_knee_margin": {
        "value": -0.10,
        "unit": "body scale",
        "description": "Hip clearly above knee at bottom. Making it more negative is more tolerant.",
    },
    "shallow_min_hip_drop": {
        "value": 0.10,
        "unit": "body scale",
        "description": "Hip-drop shallow signal. Decreasing makes the rule more tolerant.",
    },
    "heel_lift_threshold": {
        "value": 0.045,
        "unit": "body scale",
        "description": "Sustained heel rise needed for feedback. Increasing is more tolerant.",
    },
    "heel_lift_strong_threshold": {
        "value": 0.060,
        "unit": "body scale",
        "description": "Large heel rise that is reportable even when the detected bottom window is brief.",
    },
    "heel_lift_bottom_fraction": {
        "value": 0.25,
        "unit": "fraction of bottom-window frames",
        "description": "Persistence required for heel feedback. Increasing is more tolerant of brief jitter.",
    },
    "maximum_torso_lean": {
        "value": 40.0,
        "unit": "degrees from vertical",
        "description": "Absolute lean signal selected conservatively from labelled data. Increasing is more tolerant.",
    },
    "maximum_torso_change": {
        "value": 32.0,
        "unit": "degrees from calibrated standing",
        "description": "Relative lean signal. Increasing is more tolerant of squat style/body proportions.",
    },
    "chest_collapse_rise_difference": {
        "value": 0.09,
        "unit": "body scale",
        "description": "Hip rise exceeding shoulder rise during ascent. Increasing is more tolerant.",
    },
    "chest_collapse_lean_change": {
        "value": 12.0,
        "unit": "degrees",
        "description": "Required forward-tip corroboration. Increasing is more tolerant.",
    },
    "uncontrolled_rep_duration": {
        "value": 0.90,
        "unit": "seconds",
        "description": "Fast-rep signal, used only with a second signal. Decreasing is more tolerant.",
    },
    "uncontrolled_descent_duration": {
        "value": 0.32,
        "unit": "seconds",
        "description": "Fast-descent signal, used only with a second signal. Decreasing is more tolerant.",
    },
    "uncontrolled_descent_velocity": {
        "value": 430.0,
        "unit": "degrees/second",
        "description": "High descent velocity signal. Increasing is more tolerant.",
    },
    "good_probability_threshold": {
        "value": 0.65,
        "unit": "probability",
        "description": "High-confidence GOOD boundary. Decreasing labels more reps good.",
    },
    "bad_probability_threshold": {
        "value": 0.35,
        "unit": "probability of GOOD",
        "description": "High-confidence BAD boundary. Decreasing requires stronger bad confidence.",
    },
    "strong_rule_confidence": {
        "value": 0.80,
        "unit": "confidence 0-1",
        "description": "Rule confidence needed to override the model quality result. Increasing is more tolerant.",
    },
    "strong_rule_severity": {
        "value": 0.40,
        "unit": "severity 0-1",
        "description": "Rule severity needed to override the model quality result. Increasing is more tolerant.",
    },
    "minimum_feedback_rule_confidence": {
        "value": 0.70,
        "unit": "confidence 0-1",
        "description": "Minimum confidence before naming a correction. Increasing suppresses more uncertain cues.",
    },
    "minimum_feedback_rule_severity": {
        "value": 0.25,
        "unit": "severity 0-1",
        "description": "Minimum severity before naming a correction. Increasing ignores more minor errors.",
    },
    "feedback_cooldown_seconds": {
        "value": 4.0,
        "unit": "seconds",
        "description": "Minimum delay before repeating the same spoken cue. Increasing reduces repetition.",
    },
}

SQUAT_THRESHOLDS = {name: spec["value"] for name, spec in THRESHOLD_SPECS.items()}


def threshold_bundle() -> dict:
    """Return JSON/joblib-safe values and explanations for model artifacts."""
    return {"values": deepcopy(SQUAT_THRESHOLDS), "documentation": deepcopy(THRESHOLD_SPECS)}
