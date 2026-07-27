import os
import sys

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.getenv(
    "DATASET_FILE",
    os.path.join(SCRIPT_DIR, "biceps_hybrid_reps_dataset.csv"),
)
MODEL_FILE = os.getenv(
    "MODEL_FILE",
    os.path.join(SCRIPT_DIR, "bicep_curl_hybrid_quality_model.pkl"),
)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_ROWS_FOR_SPLIT = 10
FEATURE_IMPORTANCE_TOP_N = 10

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

TARGET_COLUMN = "is_good"

OLD_ERROR_COLUMNS = [
    "err_partial_rom",
    "err_too_fast",
    "err_torso_sway",
    "err_elbow_drift",
    "err_asymmetry",
    "err_shoulder_swing",
    "err_wrist_compensation",
    "err_control_loss",
]


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def can_do_stratified_split(y):
    return y.nunique() >= 2 and len(y) >= MIN_ROWS_FOR_SPLIT and y.value_counts().min() >= 2


def get_feature_importance_records(model, feature_columns):
    importance_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    return importance_frame.to_dict(orient="records")


def print_feature_importances(feature_importances, top_n=FEATURE_IMPORTANCE_TOP_N):
    print(f"\nQuality top {top_n} feature importances:")
    for item in feature_importances[:top_n]:
        print(f"  {item['feature']}: {item['importance']:.4f}")


def evaluate_quality_model(model, X_train, y_train, X_test=None, y_test=None):
    metrics = {
        "train_accuracy": round(float(model.score(X_train, y_train)), 4),
        "test_accuracy": None,
        "classification_report": None,
        "confusion_matrix": None,
        "evaluated_on_test_split": X_test is not None and y_test is not None,
    }

    print("\nQuality train accuracy:", metrics["train_accuracy"])

    if not metrics["evaluated_on_test_split"]:
        print("Quality test metrics skipped (insufficient balanced data).")
        return metrics

    y_pred = model.predict(X_test)
    metrics["test_accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
    metrics["classification_report"] = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    print("Quality test accuracy:", metrics["test_accuracy"])
    print("\nQuality classification report:")
    print(metrics["classification_report"])
    print("Quality confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def print_label_counts(df):
    print("\nQuality label counts (is_good):")
    print(df[TARGET_COLUMN].value_counts().sort_index())

    present_error_columns = [column for column in OLD_ERROR_COLUMNS if column in df.columns]
    extra_error_columns = [
        column
        for column in df.columns
        if column.startswith("err_") and column not in present_error_columns
    ]
    all_error_columns = present_error_columns + sorted(extra_error_columns)

    if not all_error_columns:
        print("\nOld error column counts: none found.")
        return

    print("\nOld error column counts (not trained in hybrid model):")
    for column in all_error_columns:
        counts = df[column].fillna(0).astype(int).value_counts().sort_index().to_dict()
        print(f"{column}: {counts}")


def main():
    df = pd.read_csv(DATASET_FILE)
    print("Dataset loaded:", DATASET_FILE)
    print("Rows before cleaning:", len(df))
    print("Model output:", MODEL_FILE)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    df = df.dropna(subset=required_columns).copy()
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    print("Rows after cleaning:", len(df))
    print("\nFeature scaling not applied: Random Forest models are tree-based.")
    print_label_counts(df)

    y_quality = df[TARGET_COLUMN]
    if y_quality.nunique() < 2:
        raise ValueError("Quality model needs both is_good classes before training.")

    X = df[FEATURE_COLUMNS]

    if can_do_stratified_split(y_quality):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_quality,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_quality,
        )
        print("\nTrain shape:", X_train.shape)
        print("Test shape:", X_test.shape)
    else:
        X_train, y_train = X, y_quality
        X_test, y_test = None, None
        print("\nNot enough balanced data for a stratified test split.")
        print("Training/evaluating on full dataset and skipping test metrics.")

    evaluation_model = build_random_forest()
    evaluation_model.fit(X_train, y_train)
    quality_metrics = evaluate_quality_model(
        evaluation_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    quality_model = build_random_forest()
    quality_model.fit(X, y_quality)
    print("\nFinal quality model trained on all cleaned rows.")

    quality_feature_importances = get_feature_importance_records(
        quality_model,
        FEATURE_COLUMNS,
    )
    print_feature_importances(quality_feature_importances)

    bundle = {
        "version": "bicep_curl_hybrid_quality_only_v1",
        "exercise": "bicep_curl",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "quality_model": quality_model,
        "quality_metrics": quality_metrics,
        "quality_feature_importances": quality_feature_importances,
        "rule_reason_engine": "frontend_or_python_rules",
        "runtime_metadata": {
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "sklearn_version": sklearn.__version__,
        },
    }

    joblib.dump(bundle, MODEL_FILE)
    print(f"\nSaved quality-only model bundle to: {MODEL_FILE}")


if __name__ == "__main__":
    main()
