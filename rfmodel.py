import os
import joblib
import pandas as pd
import sklearn
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


DATASET_FILE = os.getenv("DATASET_FILE", "reps_dataset_v2.csv")
MODEL_FILE = os.getenv("MODEL_FILE", "bicep_curl_model.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_ROWS_FOR_SPLIT = 10
FEATURE_IMPORTANCE_TOP_N = 5

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

ERROR_COLUMNS = [
    "err_partial_rom",
    "err_too_fast",
    "err_torso_sway",
    "err_asymmetry",
]

TARGET_COLUMN = "is_good"


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


def print_feature_importances(model_name, feature_importances, top_n=FEATURE_IMPORTANCE_TOP_N):
    print(f"\n{model_name} top feature importances:")
    for item in feature_importances[:top_n]:
        print(f"  {item['feature']}: {item['importance']:.4f}")


def evaluate_binary_model(model_name, model, X_train, y_train, X_test=None, y_test=None):
    metrics = {
        "train_accuracy": round(float(model.score(X_train, y_train)), 4),
        "test_accuracy": None,
        "classification_report": None,
        "confusion_matrix": None,
        "evaluated_on_test_split": X_test is not None and y_test is not None,
    }

    print(f"{model_name} train accuracy:", metrics["train_accuracy"])

    if not metrics["evaluated_on_test_split"]:
        print(f"{model_name} test metrics skipped (insufficient balanced data).")
        return metrics

    y_pred = model.predict(X_test)
    metrics["test_accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
    metrics["classification_report"] = classification_report(
        y_test,
        y_pred,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    print(f"{model_name} test accuracy:", metrics["test_accuracy"])
    print(f"\n{model_name} classification report:")
    print(metrics["classification_report"])
    print(f"{model_name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


df = pd.read_csv(DATASET_FILE)
print("Dataset loaded:", DATASET_FILE)
print("Rows:", len(df))
print("Model output:", MODEL_FILE)

required_columns = FEATURE_COLUMNS + ERROR_COLUMNS + [TARGET_COLUMN]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
df[ERROR_COLUMNS] = df[ERROR_COLUMNS].fillna(0).astype(int)
df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

print("\nFeature scaling not applied: Random Forest models are tree-based and do not require normalized inputs.")

print("\nQuality label counts (is_good):")
print(df[TARGET_COLUMN].value_counts().sort_index())

print("\nActive reason label counts:")
for col in ERROR_COLUMNS:
    counts = df[col].value_counts().sort_index().to_dict()
    print(f"{col}: {counts}")

df[FEATURE_COLUMNS + ERROR_COLUMNS + [TARGET_COLUMN]].head()


X = df[FEATURE_COLUMNS]
y_quality = df[TARGET_COLUMN]

quality_metrics = None
quality_feature_importances = []
can_train_quality = y_quality.nunique() >= 2

if can_do_stratified_split(y_quality):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_quality,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_quality,
    )
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
else:
    X_train, y_train = X, y_quality
    X_test, y_test = None, None
    print("Not enough balanced data for a stratified test split.")
    print("Training quality model on full dataset and skipping test metrics.")


quality_model = None

if can_train_quality:
    quality_model = build_random_forest()
    quality_model.fit(X_train, y_train)
    print("Quality model trained.")
else:
    print("Quality model not trained: only one class found in is_good.")


if quality_model is not None:
    quality_metrics = evaluate_binary_model(
        "Quality",
        quality_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    quality_feature_importances = get_feature_importance_records(
        quality_model,
        FEATURE_COLUMNS,
    )
    print_feature_importances("Quality", quality_feature_importances)


error_models = {}
error_metrics = {}
error_feature_importances = {}

for error_col in ERROR_COLUMNS:
    y_error = df[error_col]

    if y_error.nunique() < 2:
        constant_value = int(y_error.iloc[0])
        error_models[error_col] = {
            "type": "constant",
            "value": constant_value,
        }
        error_metrics[error_col] = {
            "type": "constant",
            "value": constant_value,
            "train_accuracy": 1.0,
            "test_accuracy": None,
            "classification_report": None,
            "confusion_matrix": None,
            "evaluated_on_test_split": False,
        }
        error_feature_importances[error_col] = []
        print(f"{error_col}: only one class ({constant_value}), saved as constant predictor.")
        continue

    if can_do_stratified_split(y_error):
        X_error_train, X_error_test, y_error_train, y_error_test = train_test_split(
            X,
            y_error,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_error,
        )
        print(f"\n{error_col}: train/test split enabled.")
    else:
        X_error_train, y_error_train = X, y_error
        X_error_test, y_error_test = None, None
        print(f"\n{error_col}: insufficient balanced data for a stratified test split.")

    model = build_random_forest()
    model.fit(X_error_train, y_error_train)

    feature_importances = get_feature_importance_records(
        model,
        FEATURE_COLUMNS,
    )
    metrics = evaluate_binary_model(
        error_col,
        model,
        X_error_train,
        y_error_train,
        X_error_test,
        y_error_test,
    )

    error_models[error_col] = {
        "type": "model",
        "model": model,
    }
    error_metrics[error_col] = metrics
    error_feature_importances[error_col] = feature_importances

    print(f"{error_col}: model trained.")
    print_feature_importances(error_col, feature_importances)


bundle = {
    "version": "v4_binary_quality_plus_active_reason_validation",
    "dataset_file": DATASET_FILE,
    "feature_columns": FEATURE_COLUMNS,
    "error_columns": ERROR_COLUMNS,
    "target_column": TARGET_COLUMN,
    "quality_model": quality_model,
    "quality_metrics": quality_metrics,
    "quality_feature_importances": quality_feature_importances,
    "error_models": error_models,
    "error_metrics": error_metrics,
    "error_feature_importances": error_feature_importances,
    "runtime_metadata": {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
    },
}

joblib.dump(bundle, MODEL_FILE)
print(f"\nSaved model bundle to: {MODEL_FILE}")
