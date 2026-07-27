@echo off
REM ===========================================================================
REM  Launcher for the standalone gui_app.py (PyQt6 single-file app) using the
REM  Python 3.12 virtual environment (required for MediaPipe mp.solutions).
REM  gui_app.py is expected one level up (Desktop root); falls back to a copy
REM  placed inside this folder.
REM ===========================================================================
cd /d "%~dp0"
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run setup_venv.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
set "GUI_APP=%~dp0..\gui_app.py"
if not exist "%GUI_APP%" set "GUI_APP=%~dp0gui_app.py"
if not exist "%GUI_APP%" (
    echo [ERROR] gui_app.py not found (expected at %~dp0..\gui_app.py).
    pause
    exit /b 1
)
python "%GUI_APP%"
