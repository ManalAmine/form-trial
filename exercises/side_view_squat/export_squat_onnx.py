"""Export the quality-only squat Random Forest and browser runtime manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

try:
    from .config import EXERCISE_ID, EXERCISE_NAME, FEATURE_COLUMNS, MODEL_FILENAME, MODEL_VERSION
    from .squat_feedback_rules import FEEDBACK
except ImportError:
    from config import EXERCISE_ID, EXERCISE_NAME, FEATURE_COLUMNS, MODEL_FILENAME, MODEL_VERSION
    from squat_feedback_rules import FEEDBACK


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = SCRIPT_DIR / "models" / MODEL_FILENAME
DEFAULT_OUTPUT = SCRIPT_DIR.parent.parent / "browser_models" / EXERCISE_ID / "v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(model_path: Path, output_dir: Path) -> tuple[Path, Path]:
    bundle = joblib.load(model_path)
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Saved model feature order does not match config.FEATURE_COLUMNS.")
    model = bundle.get("quality_model")
    if model is None:
        raise ValueError("Bundle is missing quality_model.")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "quality.onnx"
    onnx_model = convert_sklearn(
        model,
        initial_types=[("float_input", FloatTensorType([None, len(FEATURE_COLUMNS)]))],
        options={id(model): {"zipmap": False}},
        target_opset=15,
    )
    onnx.save_model(onnx_model, str(onnx_path))
    checked = onnx.load(str(onnx_path))
    onnx.checker.check_model(checked)
    input_name = checked.graph.input[0].name
    output_names = [item.name for item in checked.graph.output]

    manifest = {
        "exercise_id": EXERCISE_ID,
        "exercise_name": EXERCISE_NAME,
        "model_version": bundle.get("version", MODEL_VERSION),
        "model_file": "quality.onnx",
        "model_sha256": _sha256(onnx_path),
        "input": {"name": input_name, "dtype": "float32", "shape": [None, len(FEATURE_COLUMNS)]},
        "output_names": output_names,
        "class_order": [int(value) for value in model.classes_],
        "feature_names": list(FEATURE_COLUMNS),
        "probability_thresholds": bundle["probability_thresholds"],
        "hardcoded_rule_configuration": bundle["hardcoded_rule_thresholds"],
        "feedback_labels": FEEDBACK,
        "decision_policy": {
            "confident_good": "p_good >= high_confidence_good unless a severe high-confidence rule fires",
            "confident_bad": "p_good <= high_confidence_bad",
            "borderline": "BAD only for a strong rule; otherwise BORDERLINE",
            "unreliable": "UNSCORED before model inference",
            "maximum_feedback_messages": 1,
        },
        "feature_contract": {
            "coordinates": "MediaPipe normalized x/y; positive y points downward",
            "angles": "smaller 2-D joint angle in degrees",
            "torso_lean": "absolute shoulder-to-hip inclination from vertical in degrees",
            "distances": "normalized by calibrated shoulder-to-ankle body scale",
            "smoothing": "MediaPipe landmark smoothing plus five-frame median measurement filtering",
            "rep_phases": ["CALIBRATING", "STANDING", "DESCENDING", "BOTTOM", "ASCENDING"],
        },
        "training_date": bundle.get("training_date"),
        "evaluation_metrics": bundle.get("evaluation_metrics"),
        "dataset_metadata": bundle.get("dataset_metadata"),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Exported {onnx_path}")
    print(f"Wrote {manifest_path}")
    return onnx_path, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.model, args.output_dir)


if __name__ == "__main__":
    main()
