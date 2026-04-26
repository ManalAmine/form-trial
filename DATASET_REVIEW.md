# Dataset Review

The improved dataset is now the main dataset:

- `reps_dataset_v2.csv`: current default dataset
- `bicep_curl_model.pkl`: current default trained model

Backups were kept so the change is still reversible:

- `reps_dataset_v2_before_fast_fix_backup.csv`
- `bicep_curl_model_before_fast_fix_backup.pkl`

## Changes Promoted Into The Main Dataset

Two rows were relabeled from good to fast bad in the current `reps_dataset_v2.csv`.

- Row index `18`
  - physical line in the CSV: `20`
  - changed from `err_too_fast=0`, `is_good=1`
  - changed to `err_too_fast=1`, `is_good=0`

- Row index `43`
  - physical line in the CSV: `55`
  - changed from `err_too_fast=0`, `is_good=1`
  - changed to `err_too_fast=1`, `is_good=0`

## Borderline Rows Left Unchanged

- `17`
- `19`
- `20`
- `48`

## Noisy Or Outlier Rows Left Unchanged

- `3`
- `22`
- `31`
- `32`
- `36`
- `38`
- `42`
- `44`

## Default Commands

Train the default model:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_model.ps1
```

Run the default live predictor:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_live.ps1
```
