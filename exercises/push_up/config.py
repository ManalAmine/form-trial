"""Single source of truth for the side-view push-up data/model contract."""

from __future__ import annotations

from copy import deepcopy

EXERCISE_NAME = "push_up"
BUNDLE_VERSION = "push_up_hybrid_quality_v1"
MODEL_FILENAME = "push_up_model.pkl"
COLLECTOR_VERSION = "push_up_collector_v1"

METADATA_COLUMNS = [
    "subject_id",
    "session_id",
    "rep_index",
    "timestamp",
    "selected_side",
    "facing_direction",
    "collector_version",
]

# This order is the contract shared by collection, validation, training, model
# metadata, and live inference. Changing it requires a new bundle version.
FEATURE_COLUMNS = [
    "min_elbow_angle",
    "max_elbow_angle",
    "start_elbow_angle",
    "end_elbow_angle",
    "elbow_rom",
    "mean_elbow_angle",
    "std_elbow_angle",
    "mean_elbow_velocity",
    "max_elbow_velocity",
    "max_descent_velocity",
    "max_ascent_velocity",
    "rep_duration",
    "descent_duration",
    "ascent_duration",
    "bottom_transition_duration",
    "descent_ascent_ratio",
    "frame_count",
    "mean_body_line_angle",
    "min_body_line_angle",
    "max_body_line_angle",
    "mean_signed_hip_offset",
    "min_signed_hip_offset",
    "max_signed_hip_offset",
    "std_hip_offset",
    "hip_offset_at_top",
    "hip_offset_at_bottom",
    "hip_sag_frame_fraction",
    "hips_high_frame_fraction",
    "shoulder_vertical_travel",
    "hip_vertical_travel",
    "shoulder_hip_travel_difference",
    "hip_to_shoulder_travel_ratio",
    "shoulder_hip_movement_correlation",
    "movement_smoothness",
    "bottom_reversal_velocity_change",
    "mean_landmark_visibility",
    "min_landmark_visibility",
    "valid_frame_ratio",
]

ERROR_COLUMNS = [
    "err_shallow_depth",
    "err_incomplete_extension",
    "err_hip_sag",
    "err_hips_too_high",
    "err_uncontrolled_tempo",
]

TARGET_COLUMN = "is_good"
CSV_COLUMNS = METADATA_COLUMNS + FEATURE_COLUMNS + ERROR_COLUMNS + [TARGET_COLUMN]

FEEDBACK_LABELS = {
    "shallow_depth": "Go lower",
    "incomplete_extension": "Finish the rep",
    "hip_sag": "Keep your hips up",
    "hips_too_high": "Lower your hips",
    "uncontrolled_tempo": "Slow down and stay controlled",
    "general": "Form needs adjustment",
    "good": "Good rep",
}

# Angles are degrees, time values seconds, velocities degrees/second, and body
# offsets/travel are normalized by shoulder-to-ankle length.
THRESHOLD_SPECS = {
    "top_elbow_angle": {"value": 160.0, "description": "Elbow angle accepted as a stable top position."},
    "bottom_elbow_angle": {"value": 105.0, "description": "Elbow angle accepted as sufficient bottom depth."},
    "state_hysteresis": {"value": 8.0, "description": "Angle margin separating state transitions."},
    "stable_frames": {"value": 3, "description": "Consecutive frames required for top and major transitions."},
    "bottom_stable_frames": {"value": 2, "description": "Consecutive reversal/bottom frames required."},
    "ascent_stable_frames": {"value": 2, "description": "Consecutive rising frames required."},
    "smoothing_window_size": {"value": 5, "description": "Median smoothing window for measurements."},
    "minimum_landmark_visibility": {"value": 0.55, "description": "Minimum mean selected-side visibility."},
    "minimum_core_visibility": {"value": 0.40, "description": "Minimum visibility for each required selected-side joint."},
    "pose_loss_timeout": {"value": 8, "description": "Consecutive missing frames allowed during an attempt."},
    "minimum_valid_frame_ratio": {"value": 0.82, "description": "Minimum valid-frame fraction for a completed attempt."},
    "minimum_valid_frames": {"value": 12, "description": "Minimum valid frames for a completed attempt."},
    "reliable_visibility_mean": {"value": 0.68, "description": "Minimum mean visibility for scoring."},
    "reliable_visibility_min": {"value": 0.35, "description": "Minimum observed visibility for scoring."},
    "side_switch_visibility_margin": {"value": 0.12, "description": "Visibility advantage required to switch sides while idle."},
    "minimum_body_frame_fraction": {"value": 0.42, "description": "Minimum shoulder-to-ankle span in the frame."},
    "maximum_body_frame_fraction": {"value": 0.96, "description": "Maximum selected-side landmark span in the frame."},
    "frame_edge_margin": {"value": 0.02, "description": "Clearance required around selected-side landmarks."},
    "maximum_side_width_ratio": {"value": 0.30, "description": "Maximum apparent torso width divided by body length."},
    "minimum_horizontal_body_ratio": {"value": 0.55, "description": "Minimum horizontal component of the shoulder-to-ankle line."},
    "descent_start_angle": {"value": 150.0, "description": "Elbow angle below which a descent may start."},
    "descent_velocity_threshold": {"value": 10.0, "description": "Negative angular speed magnitude supporting descent."},
    "ascent_velocity_threshold": {"value": 10.0, "description": "Positive angular speed supporting ascent."},
    "minimum_candidate_rom": {"value": 32.0, "description": "Minimum ROM for a labelable shallow attempt."},
    "minimum_full_rom": {"value": 48.0, "description": "Minimum ROM supporting a normal bottom transition."},
    "incomplete_candidate_angle": {"value": 130.0, "description": "Minimum return angle before an incomplete-extension reversal is labelable."},
    "maximum_rep_duration": {"value": 10.0, "description": "Attempt timeout."},
    "minimum_safe_rep_duration": {"value": 0.60, "description": "Shortest scoreable repetition."},
    "shallow_depth_angle": {"value": 115.0, "description": "Minimum elbow angle indicating shallow depth."},
    "shallow_shoulder_travel": {"value": 0.075, "description": "Low normalized shoulder travel supporting shallow depth."},
    "incomplete_extension_angle": {"value": 153.0, "description": "Final elbow angle indicating incomplete extension."},
    "hip_sag_threshold": {"value": 0.045, "description": "Normalized sustained downward hip offset."},
    "hips_too_high_threshold": {"value": 0.050, "description": "Normalized sustained upward hip offset magnitude."},
    "hip_offset_persistence_fraction": {"value": 0.25, "description": "Fraction of rep frames needed for body-alignment feedback."},
    "uncontrolled_rep_duration": {"value": 0.85, "description": "Fast total-duration signal, requiring corroboration."},
    "uncontrolled_descent_duration": {"value": 0.30, "description": "Fast descent signal, requiring corroboration."},
    "uncontrolled_angular_velocity": {"value": 430.0, "description": "High peak elbow-speed signal."},
    "uncontrolled_reversal_change": {"value": 520.0, "description": "Abrupt bottom reversal signal."},
    "uncontrolled_smoothness": {"value": 2600.0, "description": "High robust angular acceleration signal."},
    "minimum_rule_confidence": {"value": 0.70, "description": "Confidence required before a rule becomes active."},
    "ml_good_threshold": {"value": 0.65, "description": "GOOD-class probability required for final GOOD."},
}

PUSH_UP_THRESHOLDS = {name: spec["value"] for name, spec in THRESHOLD_SPECS.items()}


def threshold_bundle() -> dict:
    return {"values": deepcopy(PUSH_UP_THRESHOLDS), "documentation": deepcopy(THRESHOLD_SPECS)}

