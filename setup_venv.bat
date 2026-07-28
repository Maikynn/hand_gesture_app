@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python Launcher not found. Install Python 3.12 first.
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo [1/4] Creating Python 3.12 environment on this drive...
  py -3.12 -m venv venv
  if errorlevel 1 exit /b 1
)

echo [2/4] Updating pip...
"venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [3/4] Installing application dependencies...
"venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [4/4] Installing CPU-only YOLO runtime without a duplicate OpenCV package...
"venv\Scripts\python.exe" -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1
"venv\Scripts\python.exe" -m pip install pyyaml psutil py-cpuinfo polars ultralytics-thop nvidia-ml-py
if errorlevel 1 exit /b 1
"venv\Scripts\python.exe" -m pip install --no-deps ultralytics==8.4.102
if errorlevel 1 exit /b 1

if not exist "models\piper\ru_RU-denis-medium.onnx" (
  echo Downloading the offline Russian Piper voice...
  if not exist "models\piper" mkdir "models\piper"
  "venv\Scripts\python.exe" -m piper.download_voices --download-dir "models\piper" ru_RU-denis-medium
  if errorlevel 1 exit /b 1
)

if not exist "models\whisper\tiny\model.bin" (
  echo Downloading the local Faster-Whisper tiny model...
  "venv\Scripts\hf.exe" download Systran/faster-whisper-tiny --local-dir "models\whisper\tiny"
  if errorlevel 1 exit /b 1
)

if not exist "models\silero\v5_5_ru.pt" (
  echo Downloading the high-quality Russian Silero v5.5 voice...
  if not exist "models\silero" mkdir "models\silero"
  "venv\Scripts\python.exe" -c "import torch; torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v5_5_ru.pt', 'models/silero/v5_5_ru.pt', progress=True)"
  if errorlevel 1 exit /b 1
)

echo.
echo [OK] Axi Control is ready. Start it with run.bat.
endlocal
