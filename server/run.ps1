# Запуск речевого сервера
$ErrorActionPreference = "Stop"
$venv = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (-not (Test-Path $venv)) { $venv = "python" }
& $venv -m uvicorn app:app --host 0.0.0.0 --port 8080 --app-dir $PSScriptRoot
