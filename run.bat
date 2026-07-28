@echo off
REM ===========================================================================
REM  Axi Gesture Assistant - FULL application (hand_gesture_app)
REM  Runs inside the Python 3.12 virtual environment, which is REQUIRED because
REM  Python 3.12 and MediaPipe 0.10.21 are pinned for the stable Windows
REM  mp.solutions API used by the real-time hand tracker.
REM ===========================================================================
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup_venv.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
python src/main.py
