# VITAL-PT Side-View Squat

This module follows the biceps-curl hybrid pattern while keeping one shared
implementation of geometry, repetition features, state transitions, and rules.
The Random Forest predicts only `is_good`; hard-coded rules explain one strong,
side-visible issue after a completed repetition.

The old `squat_reps_dataset.csv` in this directory is retained as legacy draft
data. Its feature schema is not compatible with this implementation. New data is
written to `data/side_squat_reps.csv`, which the collector creates on first use.

## Install

Use the same Python environment as the biceps module, then ensure these packages
are available:

```powershell
python -m pip install opencv-python mediapipe numpy pandas scikit-learn joblib
python -m pip install pytest onnx skl2onnx
```

The second line is needed for tests and browser export, not normal webcam
collection.

## Collect and label data

Webcam:

```powershell
python exercises\side_view_squat\collect_squat_data.py `
  --participant-id p001 --session-id session_001 --recording-id camera_001
```

Video:

```powershell
python exercises\side_view_squat\collect_squat_data_from_video.py `
  --source "C:\path\to\side_squat.mp4" `
  --participant-id p001 --session-id session_001 --recording-id video_001
```

Hold a comfortable upright stance until calibration finishes. The camera must
show one full side from shoulder through foot. The tracked side is locked during
each repetition.

After a complete rep:

- Click **YES - GOOD** (or press `G`) to save as good, with no error labels
- Click **BAD FORM** (or press `B`) to save as bad
- `D`: discard the pending rep
- `U`: remove the most recently saved CSV row
- `Q` or `Esc`: quit

By default these are the only two labels: GOOD and BAD. The Random Forest trains
only on `is_good`. The hard-coded rule system determines the likely error after a
completed rep; it does not need a manually selected error to operate.

If you later want optional reason notes solely for evaluating the rules, launch
with `--collect-reasons`. BAD will then open a second screen containing shallow
depth, incomplete standing, heel lift, excessive torso lean, chest collapse, and
uncontrolled tempo. These note columns are never ML targets.

The top of the collection window always shows the current phase, calibration
progress, knee and hip angles, and detected-rep count. When a full repetition is
detected, the picture freezes and displays a large `REP DETECTED` question with
the two clickable GOOD/BAD buttons. If that question never appears, the state machine did
not receive a complete stand-down-stand movement; check the phase and camera
message before labelling anything.

Use a new `session_id` or `recording_id` for each recording session. This lets
training keep similar repetitions in the same validation fold and reduces data
leakage. With about 98 repetitions, metrics will have high uncertainty; prioritize
more people and sessions over many near-identical reps from one clip.

## Train, run, and export

```powershell
python exercises\side_view_squat\train_squat_quality_model.py
python exercises\side_view_squat\predict_squat_hybrid_live.py
python exercises\side_view_squat\export_squat_onnx.py
```

Training writes `models/side_squat_hybrid_quality_v1.pkl`. It reports
out-of-fold accuracy, precision, recall, F1, a confusion matrix, and predicted
GOOD probabilities. Grouped cross-validation is used when the participant/session/
recording metadata supports it; otherwise the trainer warns that rep-level
validation may be optimistic.

Export writes `quality.onnx` and `manifest.json` to
`browser_models/side-view-squat/v1`. The manifest is the browser contract for
feature order, coordinate conventions, class order, probability thresholds,
rules, and feedback. There is no TypeScript frontend source in this repository,
so any frontend must implement that manifest contract exactly.

Debug live inference to JSON Lines:

```powershell
python exercises\side_view_squat\predict_squat_hybrid_live.py `
  --debug-log exercises\side_view_squat\debug\squat_live.jsonl --no-voice
```

The log contains frame measurements, camera reliability, phase changes, features,
model probability, every triggered rule with severity/confidence, and the final
decision reason.

## Threshold tuning after real data collection

All adjustable values and their units/strictness direction are documented in
`config.py`. Tune against held-out sessions in this order:

1. Camera visibility, framing, and side-view thresholds. Bad tracking must become
   `UNSCORED`, not a bad-form label.
2. Personal standing calibration and rep-state thresholds. Confirm that tiny bends
   never count and ordinary reps reliably complete.
3. `minimum_rep_rom`, bottom detection, and return tolerances. These control what
   counts as a rep, not whether the rep is good.
4. The shallow-depth thresholds. Review bottom windows across different body
   proportions and keep the two-signal requirement.
5. Heel-lift persistence, torso change, and chest/hip rise thresholds. Prefer false
   negatives over criticizing normal variation.
6. Uncontrolled-tempo thresholds. Keep the two-signal requirement so a merely fast
   but controlled rep is not criticized.
7. Finally calibrate the `high_confidence_good` and `high_confidence_bad`
   probability boundaries using held-out predicted probabilities. Leave the middle
   region borderline unless a strong, reliable rule fires.

Do not tune thresholds on the same repetitions used to report final performance.
The code intentionally emits only one correction and does not diagnose knee valgus,
stance width, toe angle, knee-over-toe position, symmetry, or spinal injury risk.
