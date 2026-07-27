@echo off
REM ===========================================================================
REM  Axi Gesture Assistant - FULL application (hand_gesture_app)
REM  Runs inside the Python 3.12 virtual environment, which is REQUIRED because
REM  MediaPipe 0.10.35 on Python 3.14 has no working vision API (neither
REM  mp.solutions nor mediapipe.tasks.vision). The venv uses mediapipe 0.10.14
REM  which still ships the legacy mp.solutions API.
REM ===========================================================================
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup_venv.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python src/main.py
