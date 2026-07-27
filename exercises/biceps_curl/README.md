# VITAL-PT Biceps Curl

This folder contains the current bilateral biceps-curl hybrid pipeline. A
Random Forest classifies each completed repetition as GOOD or BAD. Deterministic
rules select one clear correction for partial range, fast tempo, torso movement,
or uneven arms.

## Main files

- `collect_bicep_hybrid_data.py`: collect labelled rep-level features
- `biceps_hybrid_reps_dataset.csv`: current training dataset
- `train_bicep_hybrid_quality_model.py`: train the quality model
- `bicep_curl_hybrid_quality_model.pkl`: live Python model bundle
- `predict_bicep_hybrid_live.py`: webcam predictor and feedback UI
- `export_bicep_hybrid_quality_onnx.py`: browser model exporter
- `FRONTEND_INTEGRATION.md`: browser implementation contract

## Commands

```powershell
python exercises\biceps_curl\collect_bicep_hybrid_data.py
python exercises\biceps_curl\train_bicep_hybrid_quality_model.py
python exercises\biceps_curl\predict_bicep_hybrid_live.py
python exercises\biceps_curl\export_bicep_hybrid_quality_onnx.py
```

The live coach requires both arms, shoulders, and hips to remain visible. It
counts complete curl attempts and gives feedback once per completed rep.
