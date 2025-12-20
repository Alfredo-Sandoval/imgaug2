#!/usr/bin/env python3
"""Legacy-style benchmark runner (parity with checks/check_performance.py).

This suite exists to make performance comparisons against the historical
`checks/check_performance.py` script easier and more reproducible.

Run from repo root:
    python -m benchmarks.legacy --output benchmarks/results
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import platform as _platform
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

import imgaug2 as ia
from imgaug2 import augmenters as iaa

try:
    tqdm = importlib.import_module("tqdm").tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _iter_with_progress(n: int, *, desc: str, leave: bool = False) -> object:
    """Iterate `range(n)` with progress when available.

    Uses `tqdm` when installed. Otherwise prints a lightweight heartbeat.
    """
    if tqdm is not None:
        return tqdm(range(n), desc=desc, leave=leave, dynamic_ncols=True)

    step = max(1, n // 10)

    def _gen() -> object:
        if n <= 0:
            return
        for i in range(n):
            if i == 0 or (i + 1) % step == 0 or (i + 1) == n:
                print(f"{desc}: {i + 1}/{n}", flush=True)
            yield i

    return _gen()


def _system_info() -> dict[str, object]:
    info: dict[str, object] = {
        "platform": _platform.system(),
        "platform_release": _platform.release(),
        "architecture": _platform.machine(),
        "processor": _platform.processor(),
        "python_version": _platform.python_version(),
        "numpy_version": np.__version__,
        "imgaug2_version": ia.__version__,
        "timestamp": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
    }

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


def _unique_name(base: str, seen: dict[str, int]) -> str:
    count = seen.get(base, 0) + 1
    seen[base] = count
    return base if count == 1 else f"{base}__{count}"


def _get_rss_bytes() -> int | None:
    """Best-effort process RSS in bytes (cross-platform-ish)."""
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
        # Backwards-compat with the existing report generator which expects
        # "images_per_sec" (even when the unit isn't images).
        "images_per_sec": float(throughput_per_sec),
        # ru_maxrss is a high-water mark; keep legacy field but expose the meaning explicitly.
        "memory_delta_mb": rss_high_water_delta_mb,
        "rss_high_water_delta_mb": rss_high_water_delta_mb,
    }


def _build_legacy_augmenters() -> list[iaa.Augmenter]:
    augmenters: list[iaa.Augmenter] = [
        iaa.Identity(name="Identity"),
        iaa.Crop(px=(0, 8), name="Crop-px"),
        iaa.Crop(percent=(0, 0.1), name="Crop-percent"),
        iaa.Fliplr(0.5, name="Fliplr"),
        iaa.Flipud(0.5, name="Flipud"),
        iaa.Grayscale((0.0, 1.0), name="Grayscale"),
        iaa.GaussianBlur((0, 3.0), name="GaussianBlur"),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.1), name="AdditiveGaussianNoise"),
        iaa.Dropout((0.0, 0.1), name="Dropout"),
        iaa.Multiply((0.5, 1.5), name="Multiply"),
        iaa.ContrastNormalization(alpha=(0.5, 2.0), name="ContrastNormalization"),
        iaa.Grayscale(alpha=(0.0, 1.0), name="Grayscale"),
        iaa.ElasticTransformation(alpha=(0.5, 8.0), sigma=1.0, name="ElasticTransformation"),
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=0,
            cval=(0, 255),
            mode="constant",
            name="AffineOrder0ModeConstant",
        ),
    ]

    for order in [0, 1]:
        augmenters.append(
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_px={"x": (-16, 16), "y": (-16, 16)},
                rotate=(-45, 45),
                shear=(-16, 16),
                order=order,
                cval=(0, 255),
                mode=ia.ALL,
                name=f"AffineOrder{order}",
            )
        )

    augmenters.append(
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_px={"x": (-16, 16), "y": (-16, 16)},
            rotate=(-45, 45),
            shear=(-16, 16),
            order=ia.ALL,
            cval=(0, 255),
            mode=ia.ALL,
            name="AffineAll",
        )
    )

    # Ensure unique names (the legacy script had duplicates, e.g. Grayscale).
    seen: dict[str, int] = {}
    for aug in augmenters:
        aug.name = _unique_name(str(aug.name), seen)

    return augmenters


def _build_keypoints_batch(*, batch_size: int, seed: int) -> dict[tuple[int, int, int, int], list[Any]]:
    rng = np.random.default_rng(seed)

    points = [
        ia.Keypoint(x=int(x), y=int(y))
        for x, y in zip(
            rng.integers(0, 32, size=(20,), dtype=np.int32),
            rng.integers(0, 32, size=(20,), dtype=np.int32),
            strict=True,
        )
    ]
    kps = ia.KeypointsOnImage(points, shape=(32, 32, 3))

    def _repeat(one: ia.KeypointsOnImage) -> list[ia.KeypointsOnImage]:
        return [one.deepcopy() for _ in range(batch_size)]

    small = _repeat(kps.on((4, 4, 3)))
    medium = _repeat(kps.on((32, 32, 3)))
    large = _repeat(kps.on((256, 256, 3)))

    return {
        (batch_size, 4, 4, 3): small,
        (batch_size, 32, 32, 3): medium,
        (batch_size, 256, 256, 3): large,
    }


def _build_image_batches(*, batch_size: int, seed: int) -> dict[tuple[int, int, int, int], np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        (batch_size, 4, 4, 3): rng.integers(0, 256, size=(batch_size, 4, 4, 3), dtype=np.uint8),
        (batch_size, 32, 32, 3): rng.integers(0, 256, size=(batch_size, 32, 32, 3), dtype=np.uint8),
        (batch_size, 256, 256, 3): rng.integers(0, 256, size=(batch_size, 256, 256, 3), dtype=np.uint8),
    }


def run_legacy_benchmark(
    *,
    output_dir: Path,
    iterations: int,
    warmup: int,
    seed: int,
    batch_size: int,
    include_images: bool,
    include_keypoints: bool,
    label: str | None,
) -> dict[str, object]:
    results: dict[str, Any] = {
        "system_info": _system_info(),
        "platform": "cpu",
        "label": label or "legacy",
        "suite": "legacy_check_performance",
        "params": {
            "iterations": int(iterations),
            "warmup": int(warmup),
            "seed": int(seed),
            "batch_size": int(batch_size),
        },
        "benchmarks": {},
    }

    augmenters = _build_legacy_augmenters()
    keypoints_batches = _build_keypoints_batch(batch_size=batch_size, seed=seed)
    image_batches = _build_image_batches(batch_size=batch_size, seed=seed)

    # Calculate total benchmarks
    num_kp = len(keypoints_batches) if include_keypoints else 0
    num_img = len(image_batches) if include_images else 0
    total_tasks = len(augmenters) * (num_kp + num_img)

    if tqdm is None:
        pbar = None
        print(f"Legacy suite: {total_tasks} benchmarks", flush=True)
        aug_iter = augmenters
    else:
        pbar = tqdm(total=total_tasks, desc="Legacy", unit="bench", leave=True, dynamic_ncols=True)
        aug_iter = augmenters

    try:
        for aug in aug_iter:
            aug_name = str(aug.name)

            if include_keypoints:
                bench_name = f"Keypoints/{aug_name}"
                results["benchmarks"][bench_name] = {}
                for cfg, kps_batch in keypoints_batches.items():
                    if pbar is not None:
                        pbar.set_postfix_str(f"{bench_name} {_config_key(cfg)}")
                    else:
                        print(f"  {bench_name} {_config_key(cfg)}", flush=True)
                    ia.seed(seed)
                    results["benchmarks"][bench_name][_config_key(cfg)] = _benchmark_callable(
                        lambda aug=aug, kps_batch=kps_batch: aug.augment_keypoints(kps_batch),
                        iterations=iterations,
                        warmup=warmup,
                        batch_size=batch_size,
                        throughput_unit="keypoints_on_image",
                        desc=f"{bench_name} {_config_key(cfg)}",
                    )
                    if pbar is not None:
                        pbar.update(1)

            if include_images:
                bench_name = f"Images/{aug_name}"
                results["benchmarks"][bench_name] = {}
                for cfg, images in image_batches.items():
                    if pbar is not None:
                        pbar.set_postfix_str(f"{bench_name} {_config_key(cfg)}")
                    else:
                        print(f"  {bench_name} {_config_key(cfg)}", flush=True)
                    ia.seed(seed)
                    results["benchmarks"][bench_name][_config_key(cfg)] = _benchmark_callable(
                        lambda aug=aug, images=images: aug.augment_images(images),
                        iterations=iterations,
                        warmup=warmup,
                        batch_size=batch_size,
                        throughput_unit="images",
                        desc=f"{bench_name} {_config_key(cfg)}",
                    )
                    if pbar is not None:
                        pbar.update(1)

    finally:
        if pbar is not None:
            pbar.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.date.today().isoformat()
    out_path = output_dir / f"legacy_cpu_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-keypoints", action="store_true")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    run_legacy_benchmark(
        output_dir=args.output,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        batch_size=args.batch_size,
        include_images=not args.skip_images,
        include_keypoints=not args.skip_keypoints,
        label=(args.label.strip() or None),
    )


if __name__ == "__main__":
    main()
