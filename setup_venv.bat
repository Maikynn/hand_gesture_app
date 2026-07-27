@echo off
REM ===========================================================================
REM  Create the Python 3.12 virtual environment with a MediaPipe build that
REM  still includes the legacy mp.solutions API (required for hand/face/pose
REM  skeleton + gesture detection). Python 3.14's mediapipe 0.10.35 removed it.
REM
REM  Prerequisites: Python 3.12 installed and available as "py -3.12"
REM  (e.g. "winget install Python.Python.3.12").
REM ===========================================================================
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment with Python 3.12...
    py -3.12 -m venv venv
) else (
    echo Virtual environment already exists. Upgrading packages...
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install "mediapipe==0.10.14" PyQt6 PyQt5 requests pyttsx3 onnxruntime SpeechRecognition
echo.
echo [OK] Virtual environment ready.
echo      Run run.bat for the full app, or run_gui_app.bat for gui_app.py.
pause
