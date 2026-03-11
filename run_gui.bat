@echo off
setlocal

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment is not ready. Run setup.bat first.
    exit /b 1
)

if "%EMO_RESULTS_DIR%"=="" set "EMO_RESULTS_DIR=%USERPROFILE%\emo-results"

".venv\Scripts\python.exe" src\ui_app.py --gui --results-dir "%EMO_RESULTS_DIR%"
