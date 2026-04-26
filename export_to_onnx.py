import argparse
import json
import os
from pathlib import Path

import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


DEFAULT_MODEL_FILE = "bicep_curl_model.pkl"
DEFAULT_OUTPUT_DIR = Path("browser_models") / "biceps-curl" / "v1"
EXERCISE_ID = "biceps-curl"
MODEL_VERSION = "v1"
GOOD_PROBA_THRESHOLD = 0.50
ERROR_PROBA_THRESHOLD = 0.45

ACTIVE_ERROR_MODELS = [
    {
        "key": "err_partial_rom",
        "label": "Partial range of motion",
        "shortLabel": "partial",
        "voiceLabel": "partial range of motion",
    },
    {
        "key": "err_too_fast",
        "label": "Too fast",
        "shortLabel": "fast",
        "voiceLabel": "too fast",
    },
    {
        "key": "err_torso_sway",
        "label": "Torso sway",
        "shortLabel": "torso",
        "voiceLabel": "torso sway",
    },
    {
        "key": "err_asymmetry",
        "label": "Arm asymmetry",
        "shortLabel": "asymmetry",
        "voiceLabel": "arm asymmetry",
    },
]


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
        "labelOutputName": output_names[0],
        "probabilityOutputName": probability_output,
        "classes": classes,
        "positiveClass": 1,
        "positiveClassIndex": classes.index(1),
    }


def export_model(model, output_path, feature_count):
    onnx_model = convert_classifier(model, feature_count)
    onnx.save_model(onnx_model, output_path)
    return get_model_io_metadata(onnx_model, model)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Export the biceps curl sklearn bundle to browser-ready ONNX files."
    )
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
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

    quality_model = bundle.get("quality_model")
    if quality_model is None:
        raise ValueError("Model bundle is missing quality_model.")

    error_models = bundle.get("error_models", {})
    feature_count = len(feature_columns)

    quality_filename = "quality.onnx"
    quality_metadata = export_model(
        quality_model,
        output_dir / quality_filename,
        feature_count,
    )

    exported_error_models = []
    for error_config in ACTIVE_ERROR_MODELS:
        error_key = error_config["key"]
        model_meta = error_models.get(error_key)
        if model_meta is None:
            raise ValueError(f"Model bundle is missing {error_key}.")

        browser_meta = dict(error_config)
        browser_meta["threshold"] = ERROR_PROBA_THRESHOLD

        if model_meta.get("type") == "constant":
            browser_meta["type"] = "constant"
            browser_meta["constantValue"] = int(model_meta.get("value", 0))
            exported_error_models.append(browser_meta)
            continue

        model = model_meta.get("model")
        if model is None:
            raise ValueError(f"{error_key} is not constant and has no model.")

        filename = f"{error_key}.onnx"
        io_metadata = export_model(model, output_dir / filename, feature_count)
        browser_meta.update(
            {
                "type": "onnx",
                "file": filename,
                **io_metadata,
            }
        )
        exported_error_models.append(browser_meta)

    manifest = {
        "schemaVersion": 1,
        "exerciseId": EXERCISE_ID,
        "modelVersion": MODEL_VERSION,
        "sourceModelFile": os.fspath(model_file),
        "sourceBundleVersion": bundle.get("version"),
        "runtime": "onnxruntime-web",
        "recommendedExecutionProvider": "wasm",
        "featureColumns": feature_columns,
        "thresholds": {
            "goodProbability": GOOD_PROBA_THRESHOLD,
            "errorProbability": ERROR_PROBA_THRESHOLD,
        },
        "qualityModel": {
            "type": "onnx",
            "file": quality_filename,
            "threshold": GOOD_PROBA_THRESHOLD,
            **quality_metadata,
        },
        "errorModels": exported_error_models,
        "runtimeMetadata": bundle.get("runtime_metadata", {}),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Exported browser model package to: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Feature count: {feature_count}")
    print("Models:")
    print(f"  - {quality_filename}")
    for error_model in exported_error_models:
        if error_model.get("type") == "constant":
            print(f"  - {error_model['key']} constant={error_model['constantValue']}")
        else:
            print(f"  - {error_model['file']}")


if __name__ == "__main__":
    main()
