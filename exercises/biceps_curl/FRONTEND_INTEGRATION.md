# Bicep Curl Hybrid Frontend Notes

The hybrid bicep curl deployment path keeps ML focused on one decision:

```text
completed rep features -> quality.onnx -> GOOD or BAD
```

Reasons are no longer ONNX models. They come from the same derived angles, range of motion, timing, torso, elbow drift, visibility, and tracking features that the tracker already builds.

## Runtime Flow

```text
Camera / video frame
MediaPipe Pose detects landmarks
Rep detection tracks one full bicep curl
Feature extraction builds one feature row per completed rep
Tracking quality is checked
Invalid motion is checked
Quality ML model predicts GOOD or BAD
If BAD, rule-based reason engine selects the reason
UI shows prediction + reason
Voice feedback speaks once per completed rep
```

## Frontend Integration Checklist

- Load only `quality.onnx`.
- Do not load `err_partial_rom.onnx`, `err_too_fast.onnx`, `err_torso_sway.onnx`, or `err_asymmetry.onnx`.
- Treat `manifest.json` `featureColumns` as the source of truth for feature order.
- Keep the rule reason engine in TypeScript using the same thresholds from `manifest.json` `reasonRules.thresholds`.
- Use `reasonRules.selectionMode: severity_weighted` so the most severe visible issue wins. Do not use fixed priority where `too fast` always beats torso sway.
- Use a ready state before the active rep. Arms down means `ready`; the rep timer starts only after both elbows bend below `startMovementThreshold`.
- Compute `repDuration` from the completed rep timestamps: `timestamps[timestamps.length - 1] - timestamps[0]`. Do not use wall-clock time since the user first stood with arms down.
- Do not use peak velocity by itself to explain `too fast`; it is noisy in the current data and good reps can still have high peak velocity.
- Compute a rule-only `meanConcentricVelocity` from the average elbow angle change during the curl-up phase divided by `concentricDuration`. This is not part of the ONNX input, but it makes fast reps easier to catch.
- For `too fast`, require clear evidence: very short total duration, very short curl-up duration, very high mean velocity, or medium-short duration combined with high mean velocity. Do not let a mildly high mean velocity trigger by itself.
- Compute a rule-only `repReachedFull` flag from a stable full-top hold, not one frame. Require `minFullTopHoldFrames` consecutive frames below `fullCurlTopThreshold`; if a counted rep returns down without that stable hold, explain it as `partial range of motion`.
- Compute rule-only `armTimingDiff` and `armTimingRatio` from the left/right top-angle timestamps if possible. These are not ONNX inputs, but they help explain clear one-arm-before-the-other reps as `arm asymmetry`.
- Only use timing for asymmetry when both `armTimingDiff` and `armTimingRatio` exceed their thresholds. Small timing differences are normal live-camera noise.
- Only use elbow drift for asymmetry when one arm drifts more than the other. If both elbows move together, prefer body-swing or check-form feedback.
- Let `partial range of motion` override strongly because it is objective.
- For `too fast`, `body swing`, and `arm asymmetry`, only override when `p_good` is below `ruleOverrideGoodProbability`. Confident good predictions should not become bad because one softer rule barely fired.
- Keep model `BAD` as `BAD` when no concrete rule reason exists. In that case use `check your form`; do not convert a below-threshold model score back to `GOOD`.
- Apply `reasonRules.thresholds.minReasonSeverity` before showing a rule reason. Tiny threshold crossings should not produce spoken feedback.
- Keep INVALID for incomplete or non-curl movement. Shoulder lift or upper-arm movement should be treated as bad-form feedback, not an invalid rep.
- Run ONNX only after the tracker returns a completed rep feature row.
- Do not run ONNX every frame.
- Separate rep detection from full-depth scoring. A partial rep should still complete and be scored:
  - `downThreshold`: arms down/ready position
  - `startMovementThreshold`: actual curl movement begins and the rep timer starts
  - `minCurlBendThreshold`: enough elbow bend to count a partial curl
  - `fullCurlTopThreshold` plus `minFullTopHoldFrames`: stable full curl depth for good ROM
  - `returnThreshold`: enough return toward arms-down to finish the rep
- Check tracking quality before model inference.
- Check invalid curl motion before model inference.
- Speak feedback once per completed rep outcome.
- Avoid React state updates inside the raw camera frame loop; use refs for frame-level tracker state and only set state on meaningful UI changes.
- Stop the camera stream and MediaPipe landmarker when leaving the page.
- Do not store raw video. Store only derived features, rep count, quality score, and reason.

## Decision Flow

```text
1. If tracking quality is bad:
   prediction = "RETAKE REP"
   reason = "tracking quality"
   voice = "Retake rep. Tracking quality is too low."

2. If the movement is not a valid curl:
   prediction = "INVALID REP"
   reason = invalid reason
   voice = "Invalid rep. Complete a full curl."

3. Run quality.onnx once and compute the rule reason.
   If reason is "partial range of motion":
      prediction = "BAD"
      reason = "partial range of motion"
      voice = "Bad rep. Use full range of motion."

4. If another concrete rule reason exists and p_good < ruleOverrideGoodProbability:
      prediction = "BAD"
      reason = rule reason
      voice = "Bad rep. [reason voice label]."

5. If p_good >= 0.50:
      prediction = "GOOD"
      reason = "none"
      voice = "Good rep."

6. If p_good < 0.50:
      prediction = "BAD"
      reason = rule reason or "check your form"
      voice = "Bad rep. Check your form."
```

## Why This Is Better For Deployment

- Less model loading: the browser loads one ONNX file instead of five.
- Faster frontend: inference runs only once per completed rep.
- Less data required: the dataset only needs reliable `is_good` labels for ML.
- Easier debugging: reasons come from transparent thresholds and feature values.
- Easier to extend: future exercises can reuse the same quality-model-plus-rules pattern.
- More explainable: feedback is tied to actual angles, ROM, speed, torso movement, and elbow drift.

## Generated Files

The hybrid scripts live in:

```text
exercises/biceps_curl/train_bicep_hybrid_quality_model.py
exercises/biceps_curl/collect_bicep_hybrid_data.py
exercises/biceps_curl/predict_bicep_hybrid_live.py
exercises/biceps_curl/export_bicep_hybrid_quality_onnx.py
```

The hybrid dataset is:

```text
exercises/biceps_curl/biceps_hybrid_reps_dataset.csv
```

It intentionally stores only the 20 derived feature columns plus `is_good`.
The old full clone with `err_*` columns is kept as:

```text
exercises/biceps_curl/biceps_hybrid_reps_dataset_with_errors_backup.csv
```

The browser package is still written to:

```text
browser_models/biceps-curl/v1/
```

That package should contain `quality.onnx` and `manifest.json`. Older error ONNX files may exist as backup artifacts, but the hybrid manifest disables them with `errorModels: []`.
