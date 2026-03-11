@echo off
setlocal

python -m venv .venv
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
".venv\Scripts\python.exe" -m unittest discover -s tests -v

echo.
echo Setup finished. Start the app with run_gui.bat
