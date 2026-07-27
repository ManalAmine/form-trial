$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "Could not find .venv Python at $pythonPath"
}

& $pythonPath (Join-Path $PSScriptRoot "exercises\biceps_curl\train_bicep_hybrid_quality_model.py")
