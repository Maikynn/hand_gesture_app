@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python Launcher not found. Install Python 3.12 first.
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo [1/3] Creating Python 3.12 environment on this drive...
  py -3.12 -m venv venv
  if errorlevel 1 exit /b 1
)

echo [2/3] Updating pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [3/3] Installing application dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo.
echo [OK] Axi Control is ready. Start it with run.bat.
endlocal
