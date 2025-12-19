"""NVIDIA helpers (no GPU acceleration implied; just metadata).

This module intentionally avoids importing optional GPU libraries (e.g. CuPy),
so it stays type-checkable and runnable in CPU-only environments.
"""

from __future__ import annotations

import shutil
import subprocess


def has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def get_nvidia_smi_info() -> dict[str, object] | None:
    """Query basic GPU info via nvidia-smi (if available)."""
    if not has_nvidia_smi():
        return None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None

    line = (result.stdout or "").strip().splitlines()[:1]
    if not line:
        return None

    parts = [p.strip() for p in line[0].split(",")]
    if len(parts) < 3:
        return None

    return {
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total": parts[2],
    }


def extra_system_info() -> dict[str, object]:
    info: dict[str, object] = {}
    gpu = get_nvidia_smi_info()
    if gpu is not None:
        info["nvidia_gpu"] = gpu
    return info

