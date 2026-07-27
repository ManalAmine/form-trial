import argparse
import json
import os
from pathlib import Path

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_MODEL_FILE = SCRIPT_DIR / "bicep_curl_hybrid_quality_model.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "browser_models" / "biceps-curl" / "v1"
EXERCISE_ID = "biceps-curl"
EXERCISE_NAME = "bicep_curl"
MODEL_VERSION = "bicep_curl_hybrid_quality_only_v1"
QUALITY_FILENAME = "quality.onnx"
GOOD_PROBA_THRESHOLD = 0.50
RULE_OVERRIDE_GOOD_PROBA_THRESHOLD = 0.65
NO_RULE_GOOD_PROBA_FLOOR = GOOD_PROBA_THRESHOLD

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

RULE_THRESHOLDS = {
    "goodProbability": GOOD_PROBA_THRESHOLD,
    "ruleOverrideGoodProbability": RULE_OVERRIDE_GOOD_PROBA_THRESHOLD,
    "noRuleGoodProbabilityFloor": NO_RULE_GOOD_PROBA_FLOOR,
    "minPoseVisibilityMean": 0.65,
    "minPoseVisibilityMin": 0.35,
    "maxTrackingLostRatio": 0.15,
    "downThreshold": 160.0,
    "startMovementThreshold": 150.0,
    "minCurlBendThreshold": 130.0,
    "fullCurlTopThreshold": 65.0,
    "returnThreshold": 150.0,
    "minPartialHoldFrames": 3,
    "minFullTopHoldFrames": 3,
    "minReturnHoldFrames": 2,
    "minElbowRom": 35.0,
    "partialRomMinRom": 50.0,
    "fullTopElbowAngle": 65.0,
    "tooFastRepDuration": 1.50,
    "tooFastConcentricDuration": 0.70,
    "tooFastEccentricDuration": 0.25,
    "tooFastPeakVelocity": 320.0,
    "tooFastMeanConcentricVelocity": 220.0,
    "tooFastComboRepDuration": 1.90,
    "tooFastComboMeanConcentricVelocity": 150.0,
    "maxTorsoSway": 8.0,
    "maxTorsoLean": 22.0,
    "maxElbowRomDiff": 25.0,
    "maxElbowDrift": 0.15,
    "maxElbowDriftDiff": 0.08,
    "maxArmTimingDiff": 0.75,
    "maxArmTimingRatio": 0.55,
    "minReasonSeverity": 0.05,
}

REASON_BASE_SCORES = {
    "partial range of motion": 0.95,
    "too fast": 0.90,
    "torso sway": 0.85,
    "arm asymmetry": 0.70,
}

REASON_SCORING = {
    "peakVelocityReasonWeight": 0.0,
    "maxPeakVelocityReasonSeverity": 0.60,
}

DISPLAY_REASON_LABELS = {
    "partial range of motion": "partial range of motion",
    "too fast": "too fast",
    "torso sway": "body swing",
    "arm asymmetry": "uneven arm movement",
    "check your form": "check your form",
    "tracking quality": "camera tracking issue",
}

VOICE_REASON_LABELS = {
    "partial range of motion": "use full range of motion",
    "too fast": "slow down",
    "torso sway": "keep your body still",
    "arm asymmetry": "move both arms evenly",
    "check your form": "check your form",
    "tracking quality": "make sure your arms are visible",
}

VOICE_FEEDBACK = {
    "GOOD": "Good rep.",
    "BAD": {
        "partial range of motion": "Bad rep. Use full range of motion.",
        "too fast": "Bad rep. Slow down.",
        "torso sway": "Bad rep. Keep your body still.",
        "arm asymmetry": "Bad rep. Move both arms evenly.",
        "check your form": "Bad rep. Check your form.",
    },
    "RETAKE REP": "Retake rep. Make sure your arms are visible.",
    "INVALID REP": "Invalid rep. Complete a full curl.",
}


def convert_classifier(model, feature_count):
    initial_types = [("float_input", FloatTensorType([None, feature_count]))]
    return convert_sklearn(
        model,
        initial_types=initial_types,
        options={id(model): {"zipmap": False}},
        target_opset=12,
    )


def get_model_io_metadata(onnx_model, sklearn_model):
    output_names = [output.name for output in onnx_model.graph.output]
    input_names = [input_value.name for input_value in onnx_model.graph.input]
    classes = [int(value) for value in sklearn_model.classes_.tolist()]

    if 1 not in classes:
        raise ValueError(f"Expected class 1 in model classes, got {classes}.")

    probability_output = output_names[-1]
    if len(output_names) >= 2:
        probability_output = output_names[1]

    return {
        "inputName": input_names[0],
        "outputName": probability_output,
        "labelOutputName": output_names[0],
        "probabilityOutputName": probability_output,
        "classes": classes,
        "positiveClass": 1,
        "positiveClassIndex": classes.index(1),
    }


def export_quality_model(model, output_path, feature_count):
    onnx_model = convert_classifier(model, feature_count)
    onnx.save_model(onnx_model, output_path)
    return get_model_io_metadata(onnx_model, model)


def get_manifest_source_model_file(model_file):
    try:
        return model_file.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return model_file.name


def get_browser_runtime_metadata(bundle):
    runtime_metadata = dict(bundle.get("runtime_metadata", {}))
    runtime_metadata.pop("python_executable", None)
    return runtime_metadata


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export only the bicep curl quality model to browser-ready ONNX."
    )
    parser.add_argument("--model-file", default=str(DEFAULT_MODEL_FILE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main():
    args = build_parser().parse_args()

    model_file = Path(args.model_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(model_file)
    feature_columns = bundle.get("feature_columns")
    if not feature_columns:
        raise ValueError("Model bundle is missing feature_columns.")
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Model bundle feature order does not match the bicep hybrid contract."
        )

    quality_model = bundle.get("quality_model")
    if quality_model is None:
        raise ValueError("Model bundle is missing quality_model.")

    feature_count = len(feature_columns)
    quality_metadata = export_quality_model(
        quality_model,
        output_dir / QUALITY_FILENAME,
        feature_count,
    )

    manifest = {
        "schemaVersion": 2,
        "exerciseId": EXERCISE_ID,
        "exerciseName": EXERCISE_NAME,
        "modelVersion": MODEL_VERSION,
        "sourceModelFile": get_manifest_source_model_file(model_file),
        "sourceBundleVersion": bundle.get("version"),
        "runtime": "onnxruntime-web",
        "recommendedExecutionProvider": "wasm",
        "featureColumns": feature_columns,
        "thresholds": {
            "goodProbability": GOOD_PROBA_THRESHOLD,
            "ruleOverrideGoodProbability": RULE_OVERRIDE_GOOD_PROBA_THRESHOLD,
            "noRuleGoodProbabilityFloor": NO_RULE_GOOD_PROBA_FLOOR,
        },
        "qualityModel": {
            "type": "onnx",
            "file": QUALITY_FILENAME,
            "threshold": GOOD_PROBA_THRESHOLD,
            **quality_metadata,
        },
        "errorModels": [],
        "reasonRules": {
            "engine": "rule_based_bicep_curl_hybrid_v1",
            "selectionMode": "severity_weighted",
            "thresholds": RULE_THRESHOLDS,
            "baseScores": REASON_BASE_SCORES,
            "scoring": REASON_SCORING,
            "displayReasonLabels": DISPLAY_REASON_LABELS,
            "voiceReasonLabels": VOICE_REASON_LABELS,
            "voiceFeedback": VOICE_FEEDBACK,
        },
        "runtimeMetadata": get_browser_runtime_metadata(bundle),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Exported quality-only browser model package to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Feature count: {feature_count}")
    print("Models:")
    print(f"  - {QUALITY_FILENAME}")
    print("Error ONNX models are intentionally not exported in the hybrid package.")


if __name__ == "__main__":
    main()
