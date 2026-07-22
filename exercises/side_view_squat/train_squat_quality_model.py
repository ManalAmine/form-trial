"""Train and evaluate the single binary Random Forest squat-quality model."""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

try:
    from .config import (
        ERROR_COLUMNS,
        EXERCISE_NAME,
        FEATURE_COLUMNS,
        METADATA_COLUMNS,
        MODEL_FILENAME,
        MODEL_VERSION,
        SQUAT_THRESHOLDS,
        TARGET_COLUMN,
        threshold_bundle,
    )
except ImportError:
    from config import (
        ERROR_COLUMNS,
        EXERCISE_NAME,
        FEATURE_COLUMNS,
        METADATA_COLUMNS,
        MODEL_FILENAME,
        MODEL_VERSION,
        SQUAT_THRESHOLDS,
        TARGET_COLUMN,
        threshold_bundle,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = SCRIPT_DIR / "data" / "side_squat_reps.csv"
DEFAULT_MODEL = SCRIPT_DIR / "models" / MODEL_FILENAME
RANDOM_STATE = 42


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Collect and label squat reps first.")
    frame = pd.read_csv(path)
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame.dropna(subset=required).copy()
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    if not set(frame[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError("is_good must contain only 0 or 1.")
    if frame[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Training needs at least one GOOD and one BAD repetition.")
    return frame


def _recording_groups(frame: pd.DataFrame) -> np.ndarray | None:
    columns = [name for name in ("participant_id", "session_id", "recording_id") if name in frame]
    if not columns:
        return None
    normalized = frame[columns].fillna("").astype(str)
    groups = normalized.agg("|".join, axis=1).to_numpy()
    if len(set(groups)) < 2 or all(group.replace("|", "") == "" for group in groups):
        return None
    return groups


def _cv_strategy(y: pd.Series, groups: np.ndarray | None):
    class_counts = y.value_counts()
    if int(class_counts.min()) < 2:
        raise ValueError("Each class needs at least two repetitions for cross-validation.")
    if groups is not None:
        group_frame = pd.DataFrame({"group": groups, "label": y.to_numpy()}).drop_duplicates()
        groups_per_class = group_frame.groupby("label")["group"].nunique()
        if len(groups_per_class) == 2 and int(groups_per_class.min()) >= 2:
            splits = min(5, int(groups_per_class.min()))
            return StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE), groups, "stratified_group"
    splits = min(5, int(class_counts.min()))
    return StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE), None, "stratified_rep"


def _metrics(y_true, probabilities) -> dict:
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "predicted_good_probabilities": [round(float(value), 6) for value in probabilities],
    }


def train(dataset_path: Path, model_path: Path) -> dict:
    frame = _load_dataset(dataset_path)
    if len(frame) < 150:
        print(
            f"WARNING: only {len(frame)} labelled repetitions are available. "
            "Treat validation metrics as uncertain and collect more participants/sessions."
        )
    X = frame[FEATURE_COLUMNS].astype(float)
    y = frame[TARGET_COLUMN]
    groups = _recording_groups(frame)
    cv, cv_groups, split_name = _cv_strategy(y, groups)
    if split_name != "stratified_group":
        print("WARNING: usable participant/session/recording groups were not available; rep-level CV may be optimistic.")

    probabilities = cross_val_predict(
        build_model(), X, y, cv=cv, groups=cv_groups, method="predict_proba", n_jobs=-1
    )[:, list(sorted(y.unique())).index(1)]
    evaluation = _metrics(y, probabilities)
    evaluation.update({"method": split_name, "folds": cv.get_n_splits(), "decision_threshold": 0.5})
    print("Out-of-fold evaluation (not training accuracy):")
    print(f"  accuracy={evaluation['accuracy']:.3f} precision={evaluation['precision']:.3f} recall={evaluation['recall']:.3f} f1={evaluation['f1']:.3f}")
    print(f"  confusion_matrix [rows actual BAD/GOOD, cols predicted BAD/GOOD]={evaluation['confusion_matrix']}")

    model = build_model()
    model.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    trained_at = datetime.now(timezone.utc).isoformat()
    bundle = {
        "quality_model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "exercise_name": EXERCISE_NAME,
        "version": MODEL_VERSION,
        "training_date": trained_at,
        "evaluation_metrics": evaluation,
        "class_labels": [int(value) for value in model.classes_],
        "probability_thresholds": {
            "high_confidence_good": SQUAT_THRESHOLDS["good_probability_threshold"],
            "high_confidence_bad": SQUAT_THRESHOLDS["bad_probability_threshold"],
        },
        "hardcoded_rule_thresholds": threshold_bundle(),
        "preprocessing": None,
        "dataset_metadata": {
            "path": str(dataset_path.resolve()),
            "sha256": dataset_hash,
            "rows": len(frame),
            "class_counts": {str(key): int(value) for key, value in y.value_counts().sort_index().items()},
            "group_count": len(set(groups)) if groups is not None else 0,
            "metadata_columns_present": [name for name in METADATA_COLUMNS if name in frame],
            "reason_columns_present": [name for name in ERROR_COLUMNS if name in frame],
        },
        "runtime_metadata": {
            "python_version": platform.python_version(),
            "python_executable": os.path.abspath(sys.executable),
            "sklearn_version": sklearn.__version__,
        },
    }
    joblib.dump(bundle, model_path)
    print(f"Saved model bundle: {model_path}")
    return bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.getenv("SQUAT_DATASET", DEFAULT_DATASET)))
    parser.add_argument("--model", type=Path, default=Path(os.getenv("SQUAT_MODEL", DEFAULT_MODEL)))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args.dataset, args.model)


if __name__ == "__main__":
    main()
