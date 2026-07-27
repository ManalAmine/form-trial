import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_FILE = ROOT_DIR / "exercises" / "biceps_curl" / "bicep_curl_hybrid_quality_model.pkl"
DEFAULT_DATASET_FILE = ROOT_DIR / "exercises" / "biceps_curl" / "biceps_hybrid_reps_dataset.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "browser_models" / "biceps-curl" / "v1"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare Python sklearn predictions with exported ONNX predictions."
    )
    parser.add_argument("--model-file", default=DEFAULT_MODEL_FILE)
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser


def probability_from_sklearn(model, feature_frame, positive_class=1):
    probabilities = model.predict_proba(feature_frame)
    classes = list(model.classes_)
    if positive_class not in classes:
        raise ValueError(f"Expected class {positive_class} in {classes}.")
    return probabilities[:, classes.index(positive_class)]


def probability_from_onnx(model_path, model_manifest, features_array):
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = model_manifest["inputName"]
    probability_output = model_manifest["probabilityOutputName"]
    outputs = session.run(
        [probability_output],
        {input_name: features_array.astype(np.float32)},
    )
    probabilities = np.asarray(outputs[0])
    positive_index = int(model_manifest["positiveClassIndex"])
    return probabilities[:, positive_index]


def validate_model(name, sklearn_model, model_path, model_manifest, feature_frame, features_array):
    sklearn_probabilities = probability_from_sklearn(
        sklearn_model,
        feature_frame,
        positive_class=int(model_manifest.get("positiveClass", 1)),
    )
    onnx_probabilities = probability_from_onnx(
        model_path,
        model_manifest,
        features_array,
    )
    diff = np.abs(sklearn_probabilities - onnx_probabilities)
    max_diff = float(diff.max()) if len(diff) else 0.0
    mean_diff = float(diff.mean()) if len(diff) else 0.0
    return {
        "name": name,
        "rows": int(len(feature_frame)),
        "maxAbsDiff": max_diff,
        "meanAbsDiff": mean_diff,
    }


def main():
    args = build_parser().parse_args()

    model_file = Path(args.model_file)
    dataset_file = Path(args.dataset_file)
    model_dir = Path(args.model_dir)
    manifest_path = model_dir / "manifest.json"

    bundle = joblib.load(model_file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    feature_columns = manifest["featureColumns"]
    df = pd.read_csv(dataset_file)
    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing feature columns: {missing}")

    feature_frame = df[feature_columns].fillna(0.0).astype(np.float32)
    features_array = feature_frame.to_numpy(dtype=np.float32)

    results = []
    results.append(
        validate_model(
            "quality",
            bundle["quality_model"],
            model_dir / manifest["qualityModel"]["file"],
            manifest["qualityModel"],
            feature_frame,
            features_array,
        )
    )

    for error_manifest in manifest.get("errorModels", []):
        if "error_models" not in bundle:
            print("Hybrid quality-only bundle has no error_models; skipping error checks.")
            break

        if error_manifest.get("type") == "constant":
            print(
                f"{error_manifest['key']}: skipped ONNX check "
                f"(constant={error_manifest['constantValue']})"
            )
            continue

        error_key = error_manifest["key"]
        error_model = bundle["error_models"][error_key]["model"]
        results.append(
            validate_model(
                error_key,
                error_model,
                model_dir / error_manifest["file"],
                error_manifest,
                feature_frame,
                features_array,
            )
        )

    failed = False
    for result in results:
        status = "OK" if result["maxAbsDiff"] <= args.tolerance else "FAIL"
        print(
            f"{status} {result['name']}: rows={result['rows']} "
            f"max_abs_diff={result['maxAbsDiff']:.8f} "
            f"mean_abs_diff={result['meanAbsDiff']:.8f}"
        )
        if status == "FAIL":
            failed = True

    if failed:
        raise SystemExit(
            f"One or more ONNX exports exceeded tolerance {args.tolerance}."
        )

    print("All ONNX exports match sklearn probabilities within tolerance.")


if __name__ == "__main__":
    main()
