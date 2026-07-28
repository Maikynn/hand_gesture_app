# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller specification for Hand Gesture Application.
Run: pyinstaller hand_gesture_app.spec
"""

import os
from pathlib import Path

block_cipher = None

# Project root
project_root = Path(__file__).parent

# Data files to include
added_files = [
    (str(project_root / 'models'), 'models'),
    (str(project_root / 'third_party'), 'third_party'),
    (str(project_root / 'config.json'), '.'),
    (str(project_root / 'LICENSE'), '.'),
    (str(project_root / 'THIRD_PARTY_NOTICES.md'), '.'),
    (str(project_root / 'src' / 'ui'), 'ui'),
    (str(project_root / 'src' / 'camera'), 'camera'),
    (str(project_root / 'src' / 'hand_processing'), 'hand_processing'),
    (str(project_root / 'src' / 'assistant'), 'assistant'),
    (str(project_root / 'src' / 'utils'), 'utils'),
]

a = Analysis(
    [str(project_root / 'src' / 'main.py')],
    pathex=[str(project_root / 'src')],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'cv2',
        'mediapipe',
        'numpy',
        'PyQt5',
        'onnxruntime',
        'pyttsx3',
        'speech_recognition',
        'requests',
        'vosk',
        'pyaudio',
        'pyautogui',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='HandGestureApp',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='HandGestureApp',
)
