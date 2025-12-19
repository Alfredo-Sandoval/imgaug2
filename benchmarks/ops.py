#!/usr/bin/env python3
"""Ops-level benchmarks (CPU vs MLX).

This complements the augmenter-level benchmark suite by timing the building-block
ops directly, including the critical "keep data on device" path for MLX.

Run from repo root (Apple Silicon recommended):
    python -m benchmarks.ops --output benchmarks/results
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform as _platform
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np


def _iter_with_progress(n: int, *, desc: str, leave: bool = False) -> object:
    """Iterate `range(n)` with progress when available.

    Uses `tqdm` when installed. Otherwise prints a lightweight heartbeat.
    """
    try:
        from tqdm import tqdm  # type: ignore[import-not-found]

        return tqdm(range(n), desc=desc, leave=leave, dynamic_ncols=True)
    except Exception:
        step = max(1, n // 10)

        def _gen() -> object:
            if n <= 0:
                return
            for i in range(n):
                if i == 0 or (i + 1) % step == 0 or (i + 1) == n:
                    print(f"{desc}: {i + 1}/{n}", flush=True)
                yield i

        return _gen()


class _MlxCore(Protocol):
    def eval(self, x: object) -> None: ...

    def clip(self, x: object, a_min: float, a_max: float) -> object: ...

    def round(self, x: object) -> object: ...

    def rint(self, x: object) -> object: ...

    def floor(self, x: object) -> object: ...

    uint8: object


class _ImgaugMlx(Protocol):
    def is_available(self) -> bool: ...

    def gaussian_blur(self, images: object, *, sigma: float) -> object: ...

    def additive_gaussian_noise(self, images: object, *, scale: float) -> object: ...

    def coarse_dropout(
        self,
        images: object,
        *,
        p: float,
        size_px: int,
        per_channel: bool = False,
        seed: int | None = None,
    ) -> object: ...

    def to_device(self, images: object) -> object: ...

    def to_host(
        self,
        images: object,
        *,
        dtype: np.dtype | None = None,
        clip: bool = True,
        round: bool = True,
    ) -> np.ndarray: ...

    def fliplr(self, images: object) -> object: ...

    def flipud(self, images: object) -> object: ...

    def affine_transform(
        self,
        images: object,
        matrix_2x3: np.ndarray,
        *,
        order: int,
        mode: str,
        cval: float,
    ) -> object: ...

    def perspective_transform(
        self,
        images: object,
        matrix_3x3: np.ndarray,
        *,
        order: int,
        mode: str,
        cval: float,
    ) -> object: ...

    def chain(self, x: object, *ops: Callable[[object], object]) -> object: ...

    def add(self, x: object, value: object) -> object: ...

    def linear_contrast(self, x: object, *, factor: float) -> object: ...

    def multiply(self, x: object, *, value: float) -> object: ...


def _mx_eval(mx_local: _MlxCore, value: object) -> None:
    mx_local.eval(value)
    sync = getattr(mx_local, "synchronize", None)
    if callable(sync):
        sync()


def _system_info() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": _platform.system(),
        "platform_release": _platform.release(),
        "architecture": _platform.machine(),
        "processor": _platform.processor(),
        "python_version": _platform.python_version(),
        "numpy_version": np.__version__,
        "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }

    try:
        import imgaug2 as ia

        info["imgaug2_version"] = ia.__version__
    except Exception:
        info["imgaug2_version"] = "unknown"

    try:
        import importlib.metadata as md

        info["mlx_version"] = md.version("mlx")
    except Exception:
        pass

    try:
        import cv2

        info["opencv_version"] = cv2.__version__
        try:
            info["opencv_threads"] = int(cv2.getNumThreads())
        except Exception:
            pass
    except Exception:
        pass

    # Best-effort extra metadata (no optional deps required).
    try:
        from benchmarks.platforms import apple_silicon, nvidia_cuda

        info.update(apple_silicon.extra_system_info())
        info.update(nvidia_cuda.extra_system_info())
    except Exception:
        pass

    return info


def _config_key(cfg: tuple[int, int, int, int]) -> str:
    b, h, w, c = cfg
    return f"{b}x{h}x{w}x{c}"


def _get_rss_bytes() -> int | None:
    try:
        import resource
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


def _timing_stats(samples_s: np.ndarray) -> dict[str, float]:
    return {
        "total_time_s": float(np.sum(samples_s)),
        "avg_time_s": float(np.mean(samples_s)),
        "min_time_s": float(np.min(samples_s)),
        "max_time_s": float(np.max(samples_s)),
        "std_time_s": float(np.std(samples_s)),
        "p50_time_s": float(np.percentile(samples_s, 50)),
        "p95_time_s": float(np.percentile(samples_s, 95)),
    }


def _benchmark_callable(
    fn: Callable[[], object],
    *,
    iterations: int,
    warmup: int,
    batch_size: int,
    throughput_unit: str,
    desc: str,
) -> dict[str, float | int | None | str]:
    if iterations <= 0:
        raise ValueError("iterations must be > 0")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")

    rss_before = _get_rss_bytes()
    for _ in _iter_with_progress(warmup, desc=f"{desc} warmup", leave=False):
        fn()

    samples_s = np.empty((iterations,), dtype=np.float64)
    for i in _iter_with_progress(iterations, desc=f"{desc} timing", leave=False):
        start = time.perf_counter()
        fn()
        samples_s[i] = time.perf_counter() - start

    rss_after = _get_rss_bytes()
    stats = _timing_stats(samples_s)

    throughput_per_sec = (batch_size * iterations) / float(stats["total_time_s"])
    rss_high_water_delta_mb: float | None
    if rss_before is None or rss_after is None:
        rss_high_water_delta_mb = None
    else:
        rss_high_water_delta_mb = (rss_after - rss_before) / (1024 * 1024)

    return {
        "iterations": int(iterations),
        "warmup": int(warmup),
        **stats,
        "throughput_per_sec": float(throughput_per_sec),
        "throughput_unit": str(throughput_unit),
        # Backwards compat with old report fields.
        "images_per_sec": float(throughput_per_sec),
        # ru_maxrss is a high-water mark; keep legacy field but expose the meaning explicitly.
        "memory_delta_mb": rss_high_water_delta_mb,
        "rss_high_water_delta_mb": rss_high_water_delta_mb,
    }


def _generate_images(cfg: tuple[int, int, int, int], *, seed: int) -> np.ndarray:
    batch, h, w, c = cfg
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch, h, w, c), dtype=np.uint8)


def _compute_gaussian_blur_ksize(sigma: float) -> int:
    # Keep this in sync with imgaug2/mlx/blur.py (and imgaug2's CPU implementation).
    if sigma < 3.0:
        ksize = 3.3 * sigma
    elif sigma < 5.0:
        ksize = 2.9 * sigma
    else:
        ksize = 2.6 * sigma
    ksize = int(max(ksize, 5))
    if ksize % 2 == 0:
        ksize += 1
    return ksize


def _cv2_gaussian_blur(images: np.ndarray, *, sigma: float) -> np.ndarray:
    import cv2

    if images.ndim != 4:
        raise ValueError(f"Expected images as (N,H,W,C), got {images.shape}.")
    n = int(images.shape[0])

    ksize = _compute_gaussian_blur_ksize(float(sigma))
    kernel = (ksize, ksize)

    out = np.empty_like(images)
    for i in range(n):
        out[i] = cv2.GaussianBlur(
            images[i],
            ksize=kernel,
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )
    return out


def _cv2_affine(images: np.ndarray, *, matrix_2x3: np.ndarray, mode: str, cval: float) -> np.ndarray:
    import cv2

    if images.ndim != 4:
        raise ValueError(f"Expected images as (N,H,W,C), got {images.shape}.")
    n, h, w, _c = images.shape

    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "nearest": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "reflect_101": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
        "wrap": cv2.BORDER_WRAP,
    }
    border_mode = border_map.get(str(mode).lower(), cv2.BORDER_CONSTANT)

    out = np.empty_like(images)
    for i in range(int(n)):
        out[i] = cv2.warpAffine(
            images[i],
            matrix_2x3,
            dsize=(int(w), int(h)),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=float(cval),
        )
    return out


def _cv2_perspective(images: np.ndarray, *, matrix_3x3: np.ndarray, mode: str, cval: float) -> np.ndarray:
    import cv2

    if images.ndim != 4:
        raise ValueError(f"Expected images as (N,H,W,C), got {images.shape}.")
    n, h, w, _c = images.shape

    border_map = {
        "constant": cv2.BORDER_CONSTANT,
        "edge": cv2.BORDER_REPLICATE,
        "nearest": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
        "reflect_101": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
        "wrap": cv2.BORDER_WRAP,
    }
    border_mode = border_map.get(str(mode).lower(), cv2.BORDER_CONSTANT)

    out = np.empty_like(images)
    for i in range(int(n)):
        out[i] = cv2.warpPerspective(
            images[i],
            matrix_3x3,
            dsize=(int(w), int(h)),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
            borderValue=float(cval),
        )
    return out


def run_ops_benchmarks(
    *,
    output_dir: Path,
    iterations: int,
    warmup: int,
    seed: int,
    label: str | None,
    only_sections: set[str] | None = None,
    batches: list[int] | None = None,
    sizes: list[tuple[int, int]] | None = None,
    max_total_pixels: int | None = None,
) -> dict[str, object]:
    results: dict[str, object] = {
        "system_info": _system_info(),
        "platform": "ops",
        "label": label or "ops",
        "suite": "ops_cpu_vs_mlx",
        "params": {
            "iterations": int(iterations),
            "warmup": int(warmup),
            "seed": int(seed),
            "only_sections": sorted(only_sections) if only_sections else None,
            "batches": batches,
            "sizes": sizes,
            "max_total_pixels": max_total_pixels,
        },
        "benchmarks": {},
    }

    # Import MLX lazily so this script still runs in CPU-only envs.
    mx: _MlxCore | None
    mlx: _ImgaugMlx | None
    try:
        import mlx.core as _mx

        import imgaug2.mlx as _mlx

        mx = _mx
        mlx = _mlx
        mlx_available = bool(mlx.is_available())
    except Exception:
        mx = None
        mlx = None
        mlx_available = False

    if batches or sizes:
        use_batches = batches or [1]
        use_sizes = sizes or [(256, 256)]
        max_pixels = max_total_pixels or 24_000_000
        configs = []
        for h, w in use_sizes:
            allowed = [b for b in use_batches if int(b) * int(h) * int(w) <= max_pixels]
            if not allowed:
                allowed = [min(use_batches)]
            for b in allowed:
                configs.append((int(b), int(h), int(w), 3))
        # deterministic order
        configs = sorted(set(configs), key=lambda t: (t[1] * t[2], t[0]))
    else:
        configs = [
            (1, 64, 64, 3),
            (1, 256, 256, 3),
            (16, 256, 256, 3),
            (32, 256, 256, 3),
        ]

    sigma = 1.0
    mode = "reflect"
    cval = 0.0

    # Shared warp matrices (same for all images in a batch for reproducibility).
    def _affine_matrix(h: int, w: int) -> np.ndarray:
        import cv2

        return cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 12.0, 1.0).astype(np.float32)

    def _perspective_matrix(h: int, w: int) -> np.ndarray:
        import cv2

        src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
        dst = np.array([[1, 2], [w - 3, 1], [2, h - 4], [w - 2, h - 2]], dtype=np.float32)
        return cv2.getPerspectiveTransform(src, dst).astype(np.float32)

    benches: dict[str, object] = results["benchmarks"]  # type: ignore[assignment]

    def _want(section: str) -> bool:
        return only_sections is None or section in only_sections

    if _want("gaussian_blur"):
        # --- gaussian_blur ---
        print("  [ops] gaussian_blur...", flush=True)
        for cfg in configs:
            cfg_key = _config_key(cfg)
            print(f"    config {cfg_key}", flush=True)
            images = _generate_images(cfg, seed=seed)
            batch = int(images.shape[0])

            benches.setdefault("gaussian_blur/cpu_cv2", {})[cfg_key] = _benchmark_callable(
                lambda images=images, sigma=sigma: _cv2_gaussian_blur(images, sigma=sigma),
                iterations=iterations,
                warmup=warmup,
                batch_size=batch,
                throughput_unit="images",
                desc=f"gaussian_blur/cpu_cv2 {cfg_key}",
            )

            if mlx_available and mlx is not None and mx is not None:
                mlx_local = mlx
                mx_local = mx

                def _mlx_roundtrip_blur(
                    images: np.ndarray = images, sigma: float = sigma, mlx_local: _ImgaugMlx = mlx_local
                ) -> None:
                    x = mlx_local.to_device(images)
                    y = mlx_local.gaussian_blur(x, sigma=sigma)
                    _ = mlx_local.to_host(y, dtype=images.dtype, round=False)

                benches.setdefault("gaussian_blur/mlx_roundtrip", {})[cfg_key] = _benchmark_callable(
                    _mlx_roundtrip_blur,
                    iterations=iterations,
                    warmup=warmup,
                    batch_size=batch,
                    throughput_unit="images",
                    desc=f"gaussian_blur/mlx_roundtrip {cfg_key}",
                )

                x = mlx_local.to_device(images)
                benches.setdefault("gaussian_blur/mlx_device", {})[cfg_key] = _benchmark_callable(
                    lambda x=x, sigma=sigma, mlx_local=mlx_local, mx_local=mx_local: _mx_eval(
                        mx_local, mlx_local.gaussian_blur(x, sigma=sigma)
                    ),
                    iterations=iterations,
                    warmup=warmup,
                    batch_size=batch,
                    throughput_unit="images",
                    desc=f"gaussian_blur/mlx_device {cfg_key}",
                )

    if _want("affine_transform"):
        # --- affine_transform ---
        print("  [ops] affine_transform...", flush=True)
        for cfg in configs:
            cfg_key = _config_key(cfg)
            print(f"    config {cfg_key}", flush=True)
            images = _generate_images(cfg, seed=seed + 1)
            batch = int(images.shape[0])
            h, w = int(images.shape[1]), int(images.shape[2])
            m = _affine_matrix(h, w)

            benches.setdefault("affine_transform/cpu_cv2", {})[cfg_key] = _benchmark_callable(
                lambda images=images, m=m, mode=mode, cval=cval: _cv2_affine(
                    images, matrix_2x3=m, mode=mode, cval=cval
                ),
                iterations=iterations,
                warmup=warmup,
                batch_size=batch,
                throughput_unit="images",
                desc=f"affine_transform/cpu_cv2 {cfg_key}",
            )

            if mlx_available and mlx is not None and mx is not None:
                mlx_local = mlx
                mx_local = mx

                def _mlx_roundtrip_affine(
                    images: np.ndarray = images,
                    m: np.ndarray = m,
                    mode: str = mode,
                    cval: float = cval,
                    mlx_local: _ImgaugMlx = mlx_local,
                ) -> None:
                    x = mlx_local.to_device(images)
                    y = mlx_local.affine_transform(x, m, order=1, mode=mode, cval=cval)
                    _ = mlx_local.to_host(y, dtype=images.dtype)

                benches.setdefault("affine_transform/mlx_roundtrip", {})[cfg_key] = _benchmark_callable(
                    _mlx_roundtrip_affine,
                    iterations=iterations,
                    warmup=warmup,
                    batch_size=batch,
                    throughput_unit="images",
                    desc=f"affine_transform/mlx_roundtrip {cfg_key}",
                )

                x = mlx_local.to_device(images)
                benches.setdefault("affine_transform/mlx_device", {})[cfg_key] = _benchmark_callable(
                    lambda x=x, m=m, mode=mode, cval=cval, mlx_local=mlx_local, mx_local=mx_local: _mx_eval(
                        mx_local, mlx_local.affine_transform(x, m, order=1, mode=mode, cval=cval)
                    ),
                    iterations=iterations,
                    warmup=warmup,
                    batch_size=batch,
                    throughput_unit="images",
                    desc=f"affine_transform/mlx_device {cfg_key}",
                )

    if _want("perspective_transform"):
        # --- perspective_transform ---
        print("  [ops] perspective_transform...", flush=True)
        if batches or sizes:
            persp_cfgs = configs
        else:
            persp_cfgs = [(1, 256, 256, 3), (16, 256, 256, 3)]
        for cfg in persp_cfgs:
            cfg_key = _config_key(cfg)
            print(f"    config {cfg_key}", flush=True)
            images = _generate_images(cfg, seed=seed + 2)
            batch = int(images.shape[0])
            h, w = int(images.shape[1]), int(images.shape[2])
            mat = _perspective_matrix(h, w)

            benches.setdefault("perspective_transform/cpu_cv2", {})[cfg_key] = _benchmark_callable(
                lambda images=images, mat=mat, mode=mode, cval=cval: _cv2_perspective(
                    images, matrix_3x3=mat, mode=mode, cval=cval
                ),
                iterations=iterations,
                warmup=warmup,
                batch_size=batch,
                throughput_unit="images",
                desc=f"perspective_transform/cpu_cv2 {cfg_key}",
            )

            if mlx_available and mlx is not None and mx is not None:
                mlx_local = mlx
                mx_local = mx

                def _mlx_roundtrip_perspective(
                    images: np.ndarray = images,
                    mat: np.ndarray = mat,
                    mode: str = mode,
                    cval: float = cval,
                    mlx_local: _ImgaugMlx = mlx_local,
                ) -> None:
                    x = mlx_local.to_device(images)
                    y = mlx_local.perspective_transform(x, mat, order=1, mode=mode, cval=cval)
                    _ = mlx_local.to_host(y, dtype=images.dtype)

                benches.setdefault("perspective_transform/mlx_roundtrip", {})[cfg_key] = _benchmark_callable(
                    _mlx_roundtrip_perspective,
                    iterations=iterations,
                    warmup=warmup,
                    batch_size=batch,
                    throughput_unit="images",
                    desc=f"perspective_transform/mlx_roundtrip {cfg_key}",
                )

                x = mlx_local.to_device(images)
                benches.setdefault("perspective_transform/mlx_device", {})[cfg_key] = _benchmark_callable(
                    lambda x=x, mat=mat, mode=mode, cval=cval, mlx_local=mlx_local, mx_local=mx_local: _mx_eval(
                        mx_local, mlx_local.perspective_transform(x, mat, order=1, mode=mode, cval=cval)
                    ),
                    iterations=iterations,
                    warmup=warmup,
                    batch_size=batch,
                    throughput_unit="images",
                    desc=f"perspective_transform/mlx_device {cfg_key}",
                )

    if _want("pipeline_blur_affine_add"):
        # --- pipeline (batch-friendly ops) ---
        print("  [ops] pipeline_blur_affine_add...", flush=True)
        if batches or sizes:
            batch_for_pipeline = 16 if 16 in batches else batches[0]
            pipeline_cfg = (int(batch_for_pipeline), 256, 256, 3)
        else:
            pipeline_cfg = (16, 256, 256, 3)
        pipe_key = _config_key(pipeline_cfg)
        images = _generate_images(pipeline_cfg, seed=seed + 3)
        batch = int(images.shape[0])
        h, w = int(images.shape[1]), int(images.shape[2])
        m = _affine_matrix(h, w)

        def _cpu_pipeline() -> None:
            out = _cv2_gaussian_blur(images, sigma=sigma)
            out = _cv2_affine(out, matrix_2x3=m, mode=mode, cval=cval)
            tmp = out.astype(np.float32) + 1.0
            _ = np.clip(np.rint(tmp), 0, 255).astype(np.uint8)

        benches.setdefault("pipeline_blur_affine_add/cpu_cv2", {})[pipe_key] = _benchmark_callable(
            _cpu_pipeline,
            iterations=iterations,
            warmup=warmup,
            batch_size=batch,
            throughput_unit="images",
            desc=f"pipeline_blur_affine_add/cpu_cv2 {pipe_key}",
        )

        if mlx_available and mlx is not None and mx is not None:
            mlx_local = mlx
            mx_local = mx

            def _mx_round(x: object, *, mx_local: _MlxCore = mx_local) -> object:
                round_fn = getattr(mx_local, "round", None) or getattr(mx_local, "rint", None)
                if callable(round_fn):
                    return round_fn(x)
                floor_fn = getattr(mx_local, "floor", None)
                if callable(floor_fn):
                    return floor_fn(x + 0.5)
                return x

            def _mx_postprocess_uint8(x: object, *, mx_local: _MlxCore = mx_local) -> object:
                y = mx_local.clip(x, 0, 255)
                y = _mx_round(y, mx_local=mx_local)
                return y.astype(mx_local.uint8)

            x_dev = mlx_local.to_device(images)

            def _mlx_chain_device() -> None:
                y = mlx_local.chain(
                    x_dev,
                    lambda t, sigma=sigma, mlx_local=mlx_local: mlx_local.gaussian_blur(t, sigma=sigma),
                    lambda t, m=m, mode=mode, cval=cval, mlx_local=mlx_local: mlx_local.affine_transform(
                        t, m, order=1, mode=mode, cval=cval
                    ),
                    lambda t, mlx_local=mlx_local: mlx_local.add(t, 1.0),
                )
                y = _mx_postprocess_uint8(y, mx_local=mx_local)
                _mx_eval(mx_local, y)

            benches.setdefault("pipeline_blur_affine_add/mlx_chain", {})[pipe_key] = _benchmark_callable(
                _mlx_chain_device,
                iterations=iterations,
                warmup=warmup,
                batch_size=batch,
                throughput_unit="images",
                desc=f"pipeline_blur_affine_add/mlx_chain {pipe_key}",
            )

            def _mlx_roundtrip_pipeline() -> None:
                x = mlx_local.to_device(images)
                y = mlx_local.gaussian_blur(x, sigma=sigma)
                y = mlx_local.affine_transform(y, m, order=1, mode=mode, cval=cval)
                y = mlx_local.add(y, 1.0)
                _ = mlx_local.to_host(y, dtype=images.dtype)

            benches.setdefault("pipeline_blur_affine_add/mlx_roundtrip", {})[pipe_key] = _benchmark_callable(
                _mlx_roundtrip_pipeline,
                iterations=iterations,
                warmup=warmup,
                batch_size=batch,
                throughput_unit="images",
                desc=f"pipeline_blur_affine_add/mlx_roundtrip {pipe_key}",
            )

    if _want("mlx_op_coverage"):
        # --- mlx_op_coverage (MLX-only ops not covered by CPU-vs-MLX sections) ---
        print("  [ops] mlx_op_coverage...", flush=True)
        if not (mlx_available and mlx is not None and mx is not None):
            print("    mlx_op_coverage skipped: MLX not available", flush=True)
        else:
            mlx_local = mlx
            mx_local = mx
            from imgaug2.mlx import blur as mlx_blur
            from imgaug2.mlx import crop as mlx_crop

            coverage_cfg = (1, 128, 128, 3)
            cfg_key = _config_key(coverage_cfg)
            images_u8 = _generate_images(coverage_cfg, seed=seed + 5)
            images_f32 = images_u8.astype(np.float32) / 255.0
            image_u8 = images_u8[0]
            image_f32 = images_f32[0]

            coverage_iterations = min(iterations, 30)
            coverage_warmup = min(warmup, 2)
            per_op_max_iters: dict[str, int] = {
                "clahe": 5,
                "elastic_transform": 5,
                "equalize": 5,
                "jpeg_compression": 3,
                "median_blur": 10,
                "piecewise_affine": 5,
            }

            def _bench_mlx_op(
                name: str,
                op_fn: Callable[[object], object],
                *,
                input_kind: str,
                input_variant: str,
                output_dtype: np.dtype | None,
                device_bench: bool = True,
            ) -> None:
                if input_kind == "batch":
                    if input_variant == "f32":
                        x_np = images_f32
                    else:
                        x_np = images_u8
                    batch_size = int(x_np.shape[0])
                elif input_kind == "single":
                    if input_variant == "f32":
                        x_np = image_f32
                    else:
                        x_np = image_u8
                    batch_size = 1
                else:
                    raise ValueError(f"Unknown input_kind={input_kind!r}")

                max_iter = per_op_max_iters.get(name, coverage_iterations)
                op_iterations = min(coverage_iterations, max_iter)
                op_warmup = min(coverage_warmup, 1 if name in per_op_max_iters else coverage_warmup)

                x_dev = mlx_local.to_device(x_np)

                def _roundtrip() -> None:
                    x = mlx_local.to_device(x_np)
                    y = op_fn(x)
                    _ = mlx_local.to_host(y, dtype=output_dtype)

                def _device() -> None:
                    y = op_fn(x_dev)
                    _mx_eval(mx_local, y)

                benches.setdefault(f"{name}/mlx_roundtrip", {})[cfg_key] = _benchmark_callable(
                    _roundtrip,
                    iterations=op_iterations,
                    warmup=op_warmup,
                    batch_size=batch_size,
                    throughput_unit="images",
                    desc=f"{name}/mlx_roundtrip {cfg_key}",
                )

                if device_bench:
                    benches.setdefault(f"{name}/mlx_device", {})[cfg_key] = _benchmark_callable(
                        _device,
                        iterations=op_iterations,
                        warmup=op_warmup,
                        batch_size=batch_size,
                        throughput_unit="images",
                        desc=f"{name}/mlx_device {cfg_key}",
                    )

            # Geometric (single-image ops)
            _bench_mlx_op(
                "elastic_transform",
                lambda x, seed=seed + 11, mlx_local=mlx_local: mlx_local.elastic_transform(
                    x, alpha=5.0, sigma=1.0, seed=seed, order=1, mode="reflect", cval=0.0
                ),
                input_kind="single",
                input_variant="u8",
                output_dtype=image_u8.dtype,
            )
            _bench_mlx_op(
                "piecewise_affine",
                lambda x, seed=seed + 12, mlx_local=mlx_local: mlx_local.piecewise_affine(
                    x, scale=0.02, nb_rows=4, nb_cols=4, seed=seed, order=1, mode="reflect", cval=0.0
                ),
                input_kind="single",
                input_variant="u8",
                output_dtype=image_u8.dtype,
            )
            _bench_mlx_op(
                "resize",
                lambda x, mlx_local=mlx_local: mlx_local.resize(x, (112, 112), order=1, mode="edge"),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "crop",
                lambda x, mlx_crop=mlx_crop: mlx_crop.crop(x, 8, 8, 96, 96),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "pad",
                lambda x, mlx_local=mlx_local: mlx_local.pad(x, 8, 8, 8, 8, mode="reflect", value=0.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "random_resized_crop",
                lambda x, seed=seed + 13, mlx_local=mlx_local: mlx_local.random_resized_crop(
                    x,
                    112,
                    112,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    seed=seed,
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

            # Color
            _bench_mlx_op(
                "grayscale",
                lambda x, mlx_local=mlx_local: mlx_local.grayscale(x, alpha=1.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "rgb_shift",
                lambda x, mlx_local=mlx_local: mlx_local.rgb_shift(x, r_shift=8.0, g_shift=-4.0, b_shift=3.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "channel_shuffle",
                lambda x, mlx_local=mlx_local: mlx_local.channel_shuffle(x, order=(2, 0, 1)),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "hue_saturation_value",
                lambda x, mlx_local=mlx_local: mlx_local.hue_saturation_value(
                    x, hue_shift=10.0, saturation_scale=1.1, value_scale=1.05
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "normalize",
                lambda x, mlx_local=mlx_local: mlx_local.normalize(
                    x, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
                ),
                input_kind="batch",
                input_variant="f32",
                output_dtype=None,
            )
            _bench_mlx_op(
                "denormalize",
                lambda x, mlx_local=mlx_local: mlx_local.denormalize(
                    x, mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)
                ),
                input_kind="batch",
                input_variant="f32",
                output_dtype=None,
            )
            _bench_mlx_op(
                "to_float",
                lambda x, mlx_local=mlx_local: mlx_local.to_float(x, max_value=255.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=None,
            )
            _bench_mlx_op(
                "from_float",
                lambda x, mlx_local=mlx_local: mlx_local.from_float(x, max_value=255.0, dtype=np.uint8),
                input_kind="batch",
                input_variant="f32",
                output_dtype=np.dtype(np.uint8),
            )
            _bench_mlx_op(
                "equalize",
                lambda x, mlx_local=mlx_local: mlx_local.equalize(x),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "clahe",
                lambda x, mlx_local=mlx_local: mlx_local.clahe(x, clip_limit=2.0, tile_grid_size=(4, 4)),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "channel_dropout",
                lambda x, mlx_local=mlx_local: mlx_local.channel_dropout(x, channel_idx=0, fill_value=0.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

            # Blur
            _bench_mlx_op(
                "average_blur",
                lambda x, mlx_blur=mlx_blur: mlx_blur.average_blur(x, 3),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "median_blur",
                lambda x, mlx_blur=mlx_blur: mlx_blur.median_blur(x, 3),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "motion_blur",
                lambda x, mlx_blur=mlx_blur: mlx_blur.motion_blur(x, k=7, angle=45.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "downscale",
                lambda x, mlx_blur=mlx_blur: mlx_blur.downscale(
                    x, scale=0.5, interpolation_down="linear", interpolation_up="linear"
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

            # Compression (host roundtrip)
            _bench_mlx_op(
                "jpeg_compression",
                lambda x, mlx_local=mlx_local: mlx_local.jpeg_compression(x, quality=30),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
                device_bench=False,
            )

            # Noise
            _bench_mlx_op(
                "additive_gaussian_noise",
                lambda x, seed=seed + 21, mlx_local=mlx_local: mlx_local.additive_gaussian_noise(
                    x, scale=5.0, seed=seed
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "multiplicative_noise",
                lambda x, seed=seed + 22, mlx_local=mlx_local: mlx_local.multiplicative_noise(
                    x, scale=0.1, per_channel=False, seed=seed
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "dropout",
                lambda x, seed=seed + 23, mlx_local=mlx_local: mlx_local.dropout(x, p=0.1, seed=seed),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "coarse_dropout",
                lambda x, seed=seed + 24, mlx_local=mlx_local: mlx_local.coarse_dropout(
                    x, p=0.1, size_px=8, per_channel=False, seed=seed
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "grid_dropout",
                lambda x, seed=seed + 25, mlx_local=mlx_local: mlx_local.grid_dropout(
                    x, ratio=0.5, grid_size=(4, 4), seed=seed
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "salt_and_pepper",
                lambda x, seed=seed + 26, mlx_local=mlx_local: mlx_local.salt_and_pepper(
                    x, p=0.05, salt_ratio=0.5, seed=seed
                ),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

            # Sharpen
            _bench_mlx_op(
                "sharpen",
                lambda x, mlx_local=mlx_local: mlx_local.sharpen(x, alpha=1.0, lightness=1.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "emboss",
                lambda x, mlx_local=mlx_local: mlx_local.emboss(x, alpha=1.0, strength=1.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "unsharp_mask",
                lambda x, mlx_local=mlx_local: mlx_local.unsharp_mask(x, sigma=1.0, strength=1.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

            # Flip
            _bench_mlx_op(
                "fliplr",
                lambda x, mlx_local=mlx_local: mlx_local.fliplr(x),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "flipud",
                lambda x, mlx_local=mlx_local: mlx_local.flipud(x),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "rot90",
                lambda x, mlx_local=mlx_local: mlx_local.rot90(x, k=1),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

            # Pooling
            _bench_mlx_op(
                "avg_pool",
                lambda x, mlx_local=mlx_local: mlx_local.avg_pool(x, 2),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "max_pool",
                lambda x, mlx_local=mlx_local: mlx_local.max_pool(x, 2),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "min_pool",
                lambda x, mlx_local=mlx_local: mlx_local.min_pool(x, 2),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

            # Pointwise
            _bench_mlx_op(
                "add",
                lambda x, mlx_local=mlx_local: mlx_local.add(x, 5.0),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "multiply",
                lambda x, mlx_local=mlx_local: mlx_local.multiply(x, value=1.1),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "linear_contrast",
                lambda x, mlx_local=mlx_local: mlx_local.linear_contrast(x, factor=1.2),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "invert",
                lambda x, mlx_local=mlx_local: mlx_local.invert(x),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "solarize",
                lambda x, mlx_local=mlx_local: mlx_local.solarize(x, threshold=128),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )
            _bench_mlx_op(
                "gamma_contrast",
                lambda x, mlx_local=mlx_local: mlx_local.gamma_contrast(x, gamma=1.5),
                input_kind="batch",
                input_variant="u8",
                output_dtype=images_u8.dtype,
            )

    if _want("pose_preset"):
        # --- pose_preset ---
        print("  [ops] pose_preset...", flush=True)
        try:
            import imgaug2.augmenters as iaa
        except Exception as exc:
            print(f"    pose_preset skipped: {exc}", flush=True)
        else:
            presets = [
                "lightning_pose_dlc",
                "lightning_pose_dlc_lr",
                "lightning_pose_dlc_top_down",
                "deeplabcut_pytorch_default",
                "sleap_default",
                "mmpose_default",
            ]
            preset_cfgs = [
                (1, 384, 384, 3),
                (4, 384, 384, 3),
                (2, 1080, 1920, 3),
            ]
            preset_iterations = min(iterations, 30)
            preset_warmup = min(warmup, 3)

            def _affine_matrix_params(
                h: int,
                w: int,
                *,
                rotate: float,
                scale: float,
                shift_x: float,
                shift_y: float,
            ) -> np.ndarray:
                import cv2

                m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rotate, scale).astype(np.float32)
                m[0, 2] += shift_x
                m[1, 2] += shift_y
                return m

            def _apply_mlx_pose_preset(
                x: object,
                *,
                preset: str,
                rng: np.random.Generator,
                mlx_local: _ImgaugMlx,
            ) -> object:
                # x is expected to be MLX array already
                h = int(x.shape[1])  # type: ignore[assignment]
                w = int(x.shape[2])  # type: ignore[assignment]

                def _maybe(p: float) -> bool:
                    return bool(rng.random() < p)

                if preset in {"lightning_pose_dlc", "deeplabcut_pytorch_default"}:
                    if _maybe(0.4):
                        rot = float(rng.uniform(-20.0, 20.0))
                    else:
                        rot = 0.0
                    if _maybe(0.4):
                        scale = 1.0 + float(rng.uniform(-0.15, 0.15))
                    else:
                        scale = 1.0
                    if _maybe(0.3):
                        shift = float(rng.uniform(-0.1, 0.1))
                    else:
                        shift = 0.0
                    m = _affine_matrix_params(
                        h,
                        w,
                        rotate=rot,
                        scale=scale,
                        shift_x=shift * w,
                        shift_y=shift * h,
                    )
                    x = mlx_local.affine_transform(x, m, order=1, mode="reflect", cval=0.0)
                    if _maybe(0.5):
                        x = mlx_local.coarse_dropout(x, p=0.1, size_px=5)
                    x = mlx_local.additive_gaussian_noise(x, scale=0.01 * 255.0)
                    return x

                if preset == "lightning_pose_dlc_lr":
                    if _maybe(0.5):
                        x = mlx_local.fliplr(x)
                    return _apply_mlx_pose_preset(x, preset="lightning_pose_dlc", rng=rng, mlx_local=mlx_local)

                if preset == "lightning_pose_dlc_top_down":
                    if _maybe(0.5):
                        x = mlx_local.fliplr(x)
                    if _maybe(0.5):
                        x = mlx_local.flipud(x)
                    return _apply_mlx_pose_preset(x, preset="lightning_pose_dlc", rng=rng, mlx_local=mlx_local)

                if preset == "sleap_default":
                    if not _maybe(0.5):
                        return x
                    rot = float(rng.uniform(-180.0, 180.0))
                    scale = 1.0 + float(rng.uniform(-0.2, 0.2))
                    shift_px = float(rng.uniform(-50.0, 50.0))
                    m = _affine_matrix_params(
                        h,
                        w,
                        rotate=rot,
                        scale=scale,
                        shift_x=shift_px,
                        shift_y=shift_px,
                    )
                    x = mlx_local.affine_transform(x, m, order=1, mode="reflect", cval=0.0)
                    x = mlx_local.add(x, float(rng.uniform(-0.02 * 255.0, 0.02 * 255.0)))
                    x = mlx_local.additive_gaussian_noise(x, scale=5.0)
                    x = mlx_local.linear_contrast(x, factor=float(rng.uniform(0.9, 1.1)))
                    x = mlx_local.multiply(x, value=float(rng.uniform(0.8, 1.2)))
                    return x

                if preset == "mmpose_default":
                    if _maybe(0.5):
                        x = mlx_local.fliplr(x)
                    rot = float(rng.uniform(-80.0, 80.0))
                    scale = 1.0 + float(rng.uniform(-0.25, 0.25))
                    shift = float(rng.uniform(-0.16, 0.16))
                    m = _affine_matrix_params(
                        h,
                        w,
                        rotate=rot,
                        scale=scale,
                        shift_x=shift * w,
                        shift_y=shift * h,
                    )
                    x = mlx_local.affine_transform(x, m, order=1, mode="reflect", cval=0.0)
                    return x

                raise ValueError(f"Unknown pose preset: {preset}")

            def _safe_benchmark(
                bench_key: str,
                cfg_key: str,
                fn: Callable[[], object],
                *,
                iterations: int,
                warmup: int,
                batch_size: int,
                throughput_unit: str,
                desc: str,
            ) -> None:
                try:
                    benches.setdefault(bench_key, {})[cfg_key] = _benchmark_callable(
                        fn,
                        iterations=iterations,
                        warmup=warmup,
                        batch_size=batch_size,
                        throughput_unit=throughput_unit,
                        desc=desc,
                    )
                except Exception as exc:
                    print(f"    pose_preset skipped: {bench_key} {cfg_key}: {exc}", flush=True)

            for preset in presets:
                try:
                    aug = iaa.PosePreset(preset=preset)
                except Exception as exc:
                    print(f"    pose_preset skipped: {preset}: {exc}", flush=True)
                    continue

                for cfg in preset_cfgs:
                    cfg_key = _config_key(cfg)
                    h, w = int(cfg[1]), int(cfg[2])
                    cfg_iterations = preset_iterations
                    cfg_warmup = preset_warmup
                    if h * w >= 1_000_000:
                        cfg_iterations = min(cfg_iterations, 10)
                        cfg_warmup = min(cfg_warmup, 2)

                    print(f"    preset {preset} config {cfg_key}", flush=True)
                    images = _generate_images(cfg, seed=seed + 4)
                    batch = int(images.shape[0])

                    _safe_benchmark(
                        f"pose_preset_cpu/{preset}",
                        cfg_key,
                        lambda images=images, aug=aug: aug(images=images),
                        iterations=cfg_iterations,
                        warmup=cfg_warmup,
                        batch_size=batch,
                        throughput_unit="images",
                        desc=f"pose_preset_cpu/{preset} {cfg_key}",
                    )

                    if mlx_available and mlx is not None and mx is not None:
                        mlx_local = mlx
                        mx_local = mx

                        rng_roundtrip = np.random.default_rng(seed + 10)
                        _safe_benchmark(
                            f"pose_preset_mlx_roundtrip/{preset}",
                            cfg_key,
                            lambda images=images, rng=rng_roundtrip, preset=preset, mlx_local=mlx_local: mlx_local.to_host(
                                _apply_mlx_pose_preset(
                                    mlx_local.to_device(images), preset=preset, rng=rng, mlx_local=mlx_local
                                )
                            ),
                            iterations=cfg_iterations,
                            warmup=cfg_warmup,
                            batch_size=batch,
                            throughput_unit="images",
                            desc=f"pose_preset_mlx_roundtrip/{preset} {cfg_key}",
                        )

                        rng_device = np.random.default_rng(seed + 20)
                        x_dev = mlx_local.to_device(images)
                        _safe_benchmark(
                            f"pose_preset_mlx_device/{preset}",
                            cfg_key,
                            lambda x=x_dev, rng=rng_device, preset=preset, mlx_local=mlx_local, mx_local=mx_local: _mx_eval(
                                mx_local, _apply_mlx_pose_preset(x, preset=preset, rng=rng, mlx_local=mlx_local)
                            ),
                            iterations=cfg_iterations,
                            warmup=cfg_warmup,
                            batch_size=batch,
                            throughput_unit="images",
                            desc=f"pose_preset_mlx_device/{preset} {cfg_key}",
                        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    label_suffix = ""
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(label)).strip("-")
        if safe_label:
            label_suffix = f"_{safe_label}"

    out_path = output_dir / f"ops_{stamp}{label_suffix}.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}", flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated sections to run (gaussian_blur, affine_transform, perspective_transform, "
        "pipeline_blur_affine_add, mlx_op_coverage, pose_preset).",
    )
    parser.add_argument(
        "--batches",
        type=str,
        default="",
        help="Comma-separated batch sizes for tiered runs (overrides default configs, uses 256x256x3).",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="",
        help="Comma-separated HxW sizes (e.g., 256x256,384x384,1080x1920).",
    )
    parser.add_argument(
        "--max-total-pixels",
        type=int,
        default=24_000_000,
        help="Max total pixels per batch when using --batches/--sizes (default: 24,000,000).",
    )
    args = parser.parse_args()

    only_sections = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    batches = [int(s) for s in args.batches.split(",") if s.strip()] if args.batches else None
    sizes = None
    if args.sizes:
        size_items = []
        for part in args.sizes.split(","):
            part = part.strip().lower().replace(" ", "")
            if not part:
                continue
            if "x" not in part:
                raise ValueError(f"Invalid size '{part}', expected HxW like 256x256")
            h_str, w_str = part.split("x", 1)
            size_items.append((int(h_str), int(w_str)))
        sizes = size_items
    run_ops_benchmarks(
        output_dir=args.output,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        label=(args.label.strip() or None),
        only_sections=only_sections,
        batches=batches,
        sizes=sizes,
        max_total_pixels=args.max_total_pixels,
    )


if __name__ == "__main__":
    main()
