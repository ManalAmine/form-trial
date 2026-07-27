# Exercise Modules

Exercise-specific scripts, datasets, and model bundles live here.

## Biceps Curl

```powershell
python exercises\biceps_curl\collect_bicep_hybrid_data.py
python exercises\biceps_curl\train_bicep_hybrid_quality_model.py
python exercises\biceps_curl\predict_bicep_hybrid_live.py
python exercises\biceps_curl\export_bicep_hybrid_quality_onnx.py
```

Explicit root launchers are available for both exercises:

```powershell
.\run_biceps_live.ps1
.\run_squat_live.ps1
.\train_biceps_model.ps1
.\train_squat_model.ps1
```

## Side-View Squat

```powershell
python exercises\side_view_squat\collect_squat_data.py
python exercises\side_view_squat\collect_squat_data_from_video.py --source "C:\path\to\squat_video.mp4"
python exercises\side_view_squat\train_squat_quality_model.py
python exercises\side_view_squat\predict_squat_hybrid_live.py
python exercises\side_view_squat\export_squat_onnx.py
```

The squat Random Forest training code is kept in one canonical Python file,
`train_squat_quality_model.py`. A guided version of the same workflow is in
`train_squat_quality_model.ipynb`.

Generated browser model packages stay in the root `browser_models` folder.
See `side_view_squat/README.md` for the data schema, controls, evaluation policy,
and threshold-tuning order. The older live-prediction script name remains as a
compatibility launcher.
