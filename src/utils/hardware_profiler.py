from __future__ import annotations

import ctypes
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict

from utils.config_store import PROJECT_ROOT


CACHE_PATH = PROJECT_ROOT / "user_data" / "hardware_profile.json"


@dataclass(frozen=True)
class HardwareProfile:
    cpu_name: str
    logical_cores: int
    ram_gb: float
    gpu_name: str
    vram_gb: float
    cuda_available: bool
    score: int
    tier: str
    camera_width: int
    camera_height: int
    camera_fps: int
    inference_interval_ms: int
    yolo_interval_ms: int
    camera_model: str
    whisper_model: str
    local_llm_hint: str
    measured_at: float

    @property
    def tier_label(self) -> str:
        return {
            "eco": "ECO · слабый ПК",
            "balanced": "BALANCED · средний ПК",
            "performance": "PERFORMANCE · мощный ПК",
        }.get(self.tier, self.tier.upper())

    def summary(self) -> str:
        gpu = self.gpu_name or "встроенная/не определена"
        return (
            f"{self.tier_label} · score {self.score}/100\n"
            f"CPU: {self.cpu_name} · {self.logical_cores} потоков\n"
            f"RAM: {self.ram_gb:.1f} ГБ · GPU: {gpu}"
            + (f" · VRAM {self.vram_gb:.1f} ГБ" if self.vram_gb else "")
            + f"\nПрофиль: {self.camera_width}×{self.camera_height} @ "
            f"{self.camera_fps} FPS · {self.camera_model.upper()} · "
            f"Whisper {self.whisper_model}"
        )


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _ram_gb() -> float:
    if os.name == "nt":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / (1024**3)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return pages * size / (1024**3)
    except (AttributeError, OSError, ValueError):
        return 0.0


def _gpu_info() -> tuple[str, float]:
    if os.name != "nt":
        return "", 0.0
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        nvidia = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=4,
            creationflags=flags,
            check=False,
        )
        first = next(
            (line for line in nvidia.stdout.splitlines() if line.strip()), ""
        )
        name, separator, memory_mb = first.rpartition(",")
        if separator:
            return name.strip(), max(0.0, float(memory_mb.strip()) / 1024.0)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM | ConvertTo-Json -Compress"
    )
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            creationflags=flags,
            check=False,
        )
        data: Any = json.loads(result.stdout.strip() or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return "", 0.0
    adapters = data if isinstance(data, list) else [data]
    best_name = ""
    best_ram = 0.0
    best_rank = -1
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        name = str(adapter.get("Name", "") or "")
        try:
            ram = max(0.0, int(adapter.get("AdapterRAM", 0) or 0) / (1024**3))
        except (TypeError, ValueError):
            ram = 0.0
        lower = name.casefold()
        rank = (
            3
            if any(token in lower for token in ("nvidia", "radeon", "intel arc"))
            else 1
        )
        if rank > best_rank or (rank == best_rank and ram > best_ram):
            best_name, best_ram, best_rank = name, ram, rank
    return best_name, best_ram


def _cuda_available() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _build_profile() -> HardwareProfile:
    cores = max(1, int(os.cpu_count() or 1))
    ram = _ram_gb()
    gpu_name, vram = _gpu_info()
    cuda = _cuda_available()
    cpu_name = (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER", "")
        or "Неизвестный CPU"
    )
    dedicated = any(
        token in gpu_name.casefold() for token in ("nvidia", "radeon", "intel arc")
    )
    score = min(40, cores * 3)
    score += min(30, round(ram * 1.25))
    score += min(30, round(vram * 4)) if dedicated else 3
    if cuda:
        score += 5
    score = max(1, min(100, score))
    if ram < 7.5 or cores <= 4 or score < 38:
        tier = "eco"
        settings = (640, 360, 24, 145, 320, "rules", "tiny", "qwen3:1.7b")
    elif score >= 72 and ram >= 15.0 and cores >= 8:
        tier = "performance"
        settings = (1280, 720, 30, 60, 100, "yolo", "base", "qwen3:8b")
    else:
        tier = "balanced"
        settings = (960, 540, 30, 95, 180, "yolo", "tiny", "qwen3:4b")
    width, height, fps, interval, yolo_interval, model, whisper, llm = settings
    return HardwareProfile(
        cpu_name=cpu_name,
        logical_cores=cores,
        ram_gb=round(ram, 1),
        gpu_name=gpu_name,
        vram_gb=round(vram, 1),
        cuda_available=cuda,
        score=score,
        tier=tier,
        camera_width=width,
        camera_height=height,
        camera_fps=fps,
        inference_interval_ms=interval,
        yolo_interval_ms=yolo_interval,
        camera_model=model,
        whisper_model=whisper,
        local_llm_hint=llm,
        measured_at=time.time(),
    )


def detect_hardware(*, force: bool = False) -> HardwareProfile:
    if not force:
        try:
            data: Dict[str, Any] = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
            if time.time() - float(data.get("measured_at", 0)) < 7 * 86400:
                return HardwareProfile(**data)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    profile = _build_profile()
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = CACHE_PATH.with_suffix(".tmp")
    temp.write_text(
        json.dumps(asdict(profile), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temp.replace(CACHE_PATH)
    return profile


class AdaptiveLoadController:
    """Adjust inference cadence at runtime without rewriting user settings."""

    def __init__(self, base_interval_ms: int, tier: str):
        self.base_interval_ms = max(40, int(base_interval_ms))
        self.current_interval_ms = self.base_interval_ms
        self.minimum = 45 if tier == "performance" else 65
        self.maximum = 420 if tier == "eco" else 300
        self.target_fps = 22 if tier == "eco" else 27
        self._last_adjust = 0.0

    def update(self, fps: float, inference_ms: float, now: float) -> int:
        if now - self._last_adjust < 3.0:
            return self.current_interval_ms
        self._last_adjust = now
        overloaded = fps < self.target_fps * 0.72 or inference_ms > 240
        headroom = fps >= self.target_fps and inference_ms < 120
        if overloaded:
            self.current_interval_ms = min(
                self.maximum, round(self.current_interval_ms * 1.18 + 8)
            )
        elif headroom and self.current_interval_ms > self.base_interval_ms:
            self.current_interval_ms = max(
                self.base_interval_ms,
                self.minimum,
                round(self.current_interval_ms * 0.90),
            )
        return self.current_interval_ms
