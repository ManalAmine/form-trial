# Form Coach ONNX + Vital-PT Integration

This document explains what was done to export the biceps curl form model to
ONNX, what the exported files mean,and how the browser version was integrated
into the Vital-PT frontend.

## Goal

The original form-analysis project was a Python webcam prototype. It used:

- MediaPipe Pose for body landmarks
- rule-based biceps curl rep tracking
- a scikit-learn Random Forest model for form quality
- separate error classifiers for likely bad-form reasons

Vital-PT is a browser app, so it cannot use the Python `.pkl` model directly.
The goal was to convert the trained model into browser-loadable ONNX files and
run the full biceps curl flow inside the Vital-PT Form Coach page.

## Source And Target Projects

Source model project:

```text
C:\Users\user\OneDrive\Desktop\form trial
```

Target frontend app:

```text
C:\Users\user\OneDrive\Desktop\PersonalProject\VITAL-PT-FE
```

Vital-PT branch used:

```text
ma/feat/integratingFormAI
```

## ONNX Export Work

The Python model bundle is:

```text
bicep_curl_model.pkl
```
 
That bundle contains:

- `quality_model`: predicts whether a rep is good or bad
- `error_models`: predicts likely bad-form reasons
- `feature_columns`: the exact feature order expected by the model
- training/runtime metadata

The browser should not load this `.pkl` file. Instead, it uses ONNX files
created by:

```text
export_to_onnx.py
```

Export command:

```powershell
.\.venv\Scripts\python.exe export_to_onnx.py
```

Output folder:

```text
browser_models/biceps-curl/v1/
```

Generated files:

```text
quality.onnx
err_partial_rom.onnx
err_too_fast.onnx
err_torso_sway.onnx
err_asymmetry.onnx
manifest.json
```

### What Each ONNX File Does

`quality.onnx`

Predicts the probability that a completed biceps curl rep is good.

`err_partial_rom.onnx`

Predicts whether the rep had partial range of motion.

`err_too_fast.onnx`

Predicts whether the rep was too fast.

`err_torso_sway.onnx`

Predicts whether the torso moved too much.

`err_asymmetry.onnx`

Predicts whether the arms were asymmetric.

`manifest.json`

The browser contract for the model package. It defines:

- model version
- feature order
- thresholds
- ONNX filenames
- ONNX input/output names
- positive class index
- human labels for error reasons

## ONNX Validation

The exported ONNX files were validated against the original scikit-learn model
using:

```text
validate_onnx_exports.py
```

Validation command:

```powershell
.\.venv\Scripts\python.exe validate_onnx_exports.py
```

The validation compares Python model probabilities against ONNX Runtime
probabilities on the dataset.

Result:

```text
All ONNX exports match sklearn probabilities within tolerance.
```

The largest observed difference was around `0.00000079`, which is effectively
identical for this use case.

## Model Input Features

The model expects one completed rep, not raw video frames.

Each rep is converted into this exact 20-feature vector:

```text
min_left_angle
max_left_angle
min_right_angle
max_right_angle
rep_duration
left_rom
right_rom
elbow_rom_diff
concentric_duration
eccentric_duration
left_peak_velocity
right_peak_velocity
torso_lean_mean
torso_lean_max
torso_sway
left_elbow_drift
right_elbow_drift
pose_visibility_mean
pose_visibility_min
tracking_lost_ratio
```

Important details:

- angles are in degrees
- durations are in seconds
- elbow drift uses normalized MediaPipe landmark coordinates
- missing values are filled with `0`
- the browser must use the manifest feature order

## Browser Integration In Vital-PT

The Vital-PT frontend was updated to run the full flow in the browser.

### Dependencies Added

In Vital-PT `package.json`:

```json
"onnxruntime-web": "^1.24.3",
"@mediapipe/tasks-vision": "^0.10.34"
```

`onnxruntime-web` runs the ONNX models in the browser.

`@mediapipe/tasks-vision` runs pose detection in the browser.

### Public Assets Added

The exported model package was copied into:

```text
public/models/biceps-curl/v1/
```

This folder contains:

```text
manifest.json
quality.onnx
err_partial_rom.onnx
err_too_fast.onnx
err_torso_sway.onnx
err_asymmetry.onnx
```

ONNX Runtime Web assets were copied into:

```text
public/ort/
```

This includes both `.wasm` and `.mjs` runtime files. The `.mjs` files are
important because ONNX Runtime dynamically imports them in the browser.

MediaPipe runtime assets were copied into:

```text
public/mediapipe/tasks-vision/wasm/
```

The pose landmarker model was added here:

```text
public/models/mediapipe/pose_landmarker_lite.task
```

## New Vital-PT Source Files

These files were added under:

```text
src/lib/form-coach/
```

### `bicepsCurlOnnx.ts`

Responsible for ONNX inference.

It:

- loads `manifest.json`
- loads `quality.onnx`
- loads the four error ONNX models
- builds a `Float32Array` feature vector in manifest order
- runs quality prediction
- runs error reason prediction
- returns `GOOD` or `BAD`, confidence, and top reason

It also sets:

```ts
ort.env.wasm.wasmPaths = "/ort/";
ort.env.wasm.numThreads = 1;
```

This makes ONNX Runtime load from Vital-PT public assets.

### `bicepsCurlTracker.ts`

Responsible for browser-side biceps curl rep tracking.

It:

- reads MediaPipe pose landmarks
- computes elbow angles
- tracks rep stages
- builds the same 20 rep-level features as Python
- validates tracking quality
- validates whether the movement looked like a curl

The tracker was tuned for browser reality:

- counts based on the active arm instead of requiring both arms perfectly
- uses less strict thresholds than the Python prototype
- tolerates noisier browser pose visibility
- returns a stage such as `ready`, `down`, `curling`, `top`, `counted`, `invalid`, or `retake`

This was needed because browser MediaPipe is noisier and lower-FPS than the
original Python webcam loop.

### `poseLandmarker.ts`

Responsible for creating the MediaPipe Pose Landmarker.

It loads:

```text
/mediapipe/tasks-vision/wasm
/models/mediapipe/pose_landmarker_lite.task
```

The lite model was chosen to reduce browser workload and keep the UI responsive.

## Form Coach Page Changes

The existing simulated Form Coach page was replaced with a live browser flow:

```text
src/screens/trainee/FormCoach.tsx
```

Before:

- fake camera placeholder
- simulated feedback
- simulated rep count

After:

- requests camera access
- runs MediaPipe pose detection
- draws pose landmarks on the video
- tracks biceps curl reps
- builds ONNX feature rows
- runs model scoring
- counts valid reps
- shows form confidence
- shows `GOOD` or `BAD`
- shows the strongest error reason
- gives optional voice feedback

Only Biceps Curl is currently enabled. Other exercises remain disabled in the
selector until models are added for them.

## Runtime Flow

The browser flow is:

1. User opens `/trainee/form-coach`
2. User selects `Biceps Curl`
3. User starts session
4. Browser requests camera permission
5. MediaPipe Pose Landmarker starts
6. Each processed frame produces pose landmarks
7. The biceps curl tracker watches angles over time
8. When a rep completes, it builds the 20-feature row
9. The app counts the valid rep
10. ONNX Runtime scores the rep
11. The UI shows quality and reason

## Performance Tuning

The first browser version felt slow, so the live loop was tuned.

Current choices:

```ts
const CAMERA_WIDTH = 480;
const CAMERA_HEIGHT = 360;
const POSE_FRAME_INTERVAL_MS = 50;
```

That means:

- request a smaller camera stream
- keep the visible video smooth
- run pose detection at about 20 FPS instead of every animation frame
- reduce the chance that MediaPipe blocks the UI

The UI also throttles state updates so React does not re-render too often during
the camera loop.

## Rep Counting Tuning

The original Python logic expected both arms to behave cleanly. In the browser,
that caused missed reps if one wrist or elbow was briefly hidden.

The browser tracker now:

- starts when at least one arm is extended enough
- marks curling when the active arm bends enough
- counts when the active arm returns down
- still records both arms for model scoring

This makes the counter more forgiving while preserving the model's ability to
detect asymmetry.

## Error Handling Improvements

The first integration only counted reps after ONNX scoring succeeded. That meant
if model inference failed, reps stayed at `0`.

This was changed.

Now:

- valid reps count immediately
- model scoring happens after the count
- if ONNX scoring fails, the rep still remains counted
- the UI shows the real scoring error message

Example:

```text
Rep counted. Model scoring failed: <actual error>
```

Error reason models also fail softly. If one error model fails, it logs a warning
and continues instead of crashing the whole rep result.

## Common Issues

### Camera Access Is Not Available

Usually happens when opening the app inside VS Code's embedded browser.

Fix:

Open the app directly in Chrome or Edge:

```text
http://localhost:3000/trainee/form-coach
```

Also check Windows camera permissions.

### Missing ONNX Runtime `.mjs`

Error example:

```text
Failed to fetch dynamically imported module: /ort/ort-wasm-simd-threaded.jsep.mjs
```

Cause:

Only `.wasm` files were copied, but ONNX Runtime also needs `.mjs` loader files.

Fix:

Ensure `public/ort/` contains both `.wasm` and `.mjs` files from:

```text
node_modules/onnxruntime-web/dist/
```

### Reps Not Counting

Watch the `Stage` badge.

If it stays on `ready`, the tracker is not seeing an arm down/start position.

If it reaches `down` but not `curling`, the elbow is not bending enough in the
pose data.

If it reaches `curling` but does not count, the tracker is not seeing the arm
return down.

Best test setup:

- stand farther from the camera
- keep shoulders, elbows, wrists, and hips visible
- use good lighting
- curl slowly at first

### Landmarks Feel Slow

Pose detection is CPU-heavy in the browser. Current tuning uses lower camera
resolution and throttled pose inference. If it still feels slow, possible next
steps are:

- lower camera resolution again
- use fewer drawn landmarks
- run pose detection in a Web Worker
- move ONNX scoring off the main UI path

## Commands Used

Export ONNX:

```powershell
.\.venv\Scripts\python.exe export_to_onnx.py
```

Validate ONNX:

```powershell
.\.venv\Scripts\python.exe validate_onnx_exports.py
```

Install frontend dependencies:

```powershell
npm install onnxruntime-web @mediapipe/tasks-vision
```

Run Vital-PT:

```powershell
npm run dev
```

Build Vital-PT:

```powershell
npm run build
```

## Verification Completed

The following checks passed after integration:

```text
ONNX validation against sklearn: passed
targeted ESLint for form-coach files: passed
npm run build: passed
```

The full project TypeScript check had pre-existing unrelated errors in older
files, but the production build completed successfully.

## Current Limitations

- Only biceps curl is supported.
- The model was trained on a small dataset.
- Browser pose tracking is sensitive to lighting, framing, and camera angle.
- The model judges completed reps; it does not classify every frame.
- The current app does not store raw video.
- The current app does not persist form-session results to the backend.

## Updating The Model Later

When retraining the Python model:

1. Retrain and save `bicep_curl_model.pkl`
2. Run `export_to_onnx.py`
3. Run `validate_onnx_exports.py`
4. Copy the new files from:

```text
browser_models/biceps-curl/v1/
```

to:

```text
VITAL-PT-FE/public/models/biceps-curl/v1/
```

5. Restart the Vital-PT dev server
6. Test `/trainee/form-coach`

If feature names or feature order change, update both:

- Python feature extraction
- browser `bicepsCurlTracker.ts`

The manifest feature order is the source of truth for ONNX inference.
