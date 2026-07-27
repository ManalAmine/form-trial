# VITAL-PT Form Coaches

This repository contains two webcam exercise coaches:

- bilateral biceps curl
- side-view squat

Each exercise keeps its collection, training, live prediction, model, and
documentation files under `exercises/`. Browser-ready ONNX packages live under
`browser_models/`.

## Setup

Create and activate a Python environment, then install the runtime packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install opencv-python mediapipe numpy pandas scikit-learn joblib
```

Tests and ONNX export also require:

```powershell
python -m pip install pytest onnx onnxruntime skl2onnx
```

## Run the live coaches

```powershell
.\run_biceps_live.ps1
.\run_squat_live.ps1
```

Equivalent direct commands:

```powershell
python exercises\biceps_curl\predict_bicep_hybrid_live.py
python exercises\side_view_squat\predict_squat_hybrid_live.py
```

## Train the models

```powershell
.\train_biceps_model.ps1
.\train_squat_model.ps1
```

## Tests

```powershell
python -m pytest tests -q
```

See [exercises/README.md](exercises/README.md) and the README inside each
exercise folder for collection controls, data schemas, and model details.
