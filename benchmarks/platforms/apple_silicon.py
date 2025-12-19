"""Apple Silicon helpers (no acceleration implied; just metadata)."""

from __future__ import annotations

import platform
import subprocess
from typing import Any


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_apple_silicon() -> bool:
    return is_macos() and platform.machine() == "arm64"


def get_chip_info() -> str | None:
    if not is_macos():
        return None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    value = (result.stdout or "").strip()
    return value or None


def numpy_uses_accelerate() -> bool | None:
    """Return True/False if we can detect Accelerate/vecLib usage; else None."""
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel
    except Exception:
        return None

    try:
        cfg: Any = np.show_config(mode="dicts")  # numpy >=1.26
    except Exception:
        return None

    blob = str(cfg).lower()
    return ("accelerate" in blob) or ("veclib" in blob)


def extra_system_info() -> dict[str, object]:
    """Extra metadata injected into the benchmark JSON on macOS."""
    info: dict[str, object] = {}
    chip = get_chip_info()
    if chip is not None:
        info["apple_chip"] = chip
    uses_acc = numpy_uses_accelerate()
    if uses_acc is not None:
        info["numpy_uses_accelerate"] = uses_acc
    return info

