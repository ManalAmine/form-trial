$pythonPath = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "Could not find .venv Python at $pythonPath"
}

& $pythonPath (Join-Path $PSScriptRoot "exercises\side_view_squat\predict_squat_hybrid_live.py")
