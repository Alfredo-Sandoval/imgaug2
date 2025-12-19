"""CPU benchmark implementation (portable; no optional deps)."""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


def _iter_with_progress(
    n: int,
    *,
    desc: str,
    leave: bool = False,
) -> object:
    """Iterate `range(n)` with progress when available.

    Uses `tqdm` when installed. Otherwise prints a lightweight heartbeat.
    """
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        return tqdm(range(n), desc=desc, leave=leave, dynamic_ncols=True)
    except Exception:
        # Heartbeat fallback: emit ~10 updates.
        step = max(1, n // 10)

        def _gen() -> object:
            if n <= 0:
                return
            for i in range(n):
                if i == 0 or (i + 1) % step == 0 or (i + 1) == n:
                    print(f"{desc}: {i + 1}/{n}", flush=True)
                yield i

        return _gen()


@dataclass(frozen=True, slots=True)
class TimingStats:
    total_s: float
    avg_s: float
    min_s: float
    max_s: float
    std_s: float
    p50_s: float
    p95_s: float


def _get_rss_bytes() -> int | None:
    """Best-effort process RSS in bytes (cross-platform-ish).

    - macOS/Linux: uses `resource` when available (peak RSS isn't "current" RSS).
    - Windows: returns None.
    """
    try:
        import resource  # pylint: disable=import-outside-toplevel
    except Exception:
        return None

    usage = resource.getrusage(resource.RUSAGE_SELF)
    maxrss = int(getattr(usage, "ru_maxrss", 0))
    if maxrss <= 0:
        return None

    # Linux: KiB, macOS: bytes
    if sys.platform == "darwin":
        return maxrss
    return maxrss * 1024


def _timing_stats(samples_s: np.ndarray) -> TimingStats:
    return TimingStats(
        total_s=float(np.sum(samples_s)),
        avg_s=float(np.mean(samples_s)),
        min_s=float(np.min(samples_s)),
        max_s=float(np.max(samples_s)),
        std_s=float(np.std(samples_s)),
        p50_s=float(np.percentile(samples_s, 50)),
        p95_s=float(np.percentile(samples_s, 95)),
    )


def benchmark_augmenter(
    aug: Any,  # noqa: ANN401
    images: np.ndarray,
    *,
    iterations: int,
    warmup: int,
    progress_desc: str | None = None,
) -> dict[str, float | int | None]:
    """Run a benchmark for a single augmenter on a pre-generated image batch."""
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    gc.collect()
    rss_before = _get_rss_bytes()

    warmup_desc = f"{progress_desc} warmup" if progress_desc else "warmup"
    timing_desc = f"{progress_desc} timing" if progress_desc else "timing"

    for _ in _iter_with_progress(warmup, desc=warmup_desc, leave=False):
        _ = aug(images=images)

    times_s: list[float] = []
    for _ in _iter_with_progress(iterations, desc=timing_desc, leave=False):
        start = time.perf_counter()
        _ = aug(images=images)
        times_s.append(time.perf_counter() - start)

    rss_after = _get_rss_bytes()
    stats = _timing_stats(np.asarray(times_s, dtype=np.float64))
    images_per_sec = (images.shape[0] * iterations) / stats.total_s

    rss_high_water_delta_mb: float | None
    if rss_before is None or rss_after is None:
        rss_high_water_delta_mb = None
    else:
        rss_high_water_delta_mb = (rss_after - rss_before) / (1024 * 1024)

    return {
        "iterations": int(iterations),
        "warmup": int(warmup),
        "total_time_s": stats.total_s,
        "avg_time_s": stats.avg_s,
        "min_time_s": stats.min_s,
        "max_time_s": stats.max_s,
        "std_time_s": stats.std_s,
        "p50_time_s": stats.p50_s,
        "p95_time_s": stats.p95_s,
        "images_per_sec": float(images_per_sec),
        # ru_maxrss is a high-water mark; keep legacy field but expose the meaning explicitly.
        "memory_delta_mb": rss_high_water_delta_mb,
        "rss_high_water_delta_mb": rss_high_water_delta_mb,
    }
