#!/usr/bin/env python3
"""imgaug2 benchmark runner.

Run from the repository root:
    python -m benchmarks.runner --platform cpu
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import platform as _platform
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

import imgaug2 as ia
from benchmarks.config import AUGMENTERS, BENCHMARK_CONFIGS
from benchmarks.platforms import apple_silicon, cpu, nvidia_cuda


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
    info.update(apple_silicon.extra_system_info())
    info.update(nvidia_cuda.extra_system_info())
    return info


def _config_key(cfg: tuple[int, int, int, int]) -> str:
    b, h, w, c = cfg
    return f"{b}x{h}x{w}x{c}"


def _generate_images(cfg: tuple[int, int, int, int], *, seed: int) -> np.ndarray:
    batch, h, w, c = cfg
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch, h, w, c), dtype=np.uint8)


def run_benchmarks(
    *,
    platform_name: str,
    output_dir: Path,
    iterations: int,
    warmup: int,
    seed: int,
    augmenter_filter: set[str] | None,
    config_filter: set[str] | None,
    fail_fast: bool,
    label: str | None,
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "system_info": _system_info(),
        "platform": platform_name,
        "label": label or platform_name,
        "params": {
            "iterations": iterations,
            "warmup": warmup,
            "seed": seed,
        },
        "benchmarks": {},
    }

    if platform_name != "cpu":
        # Today this is effectively only metadata; we keep the CLI stable for
        # future backend work.
        results["note"] = "imgaug2 runs on CPU; platform selection affects only metadata."

    # Filter augmenters first
    aug_items = [
        (name, factory)
        for name, factory in AUGMENTERS.items()
        if augmenter_filter is None or name in augmenter_filter
    ]
    total_tasks = len(aug_items) * len(
        [c for c in BENCHMARK_CONFIGS if config_filter is None or _config_key(c) in config_filter]
    )

    case_idx = 0
    if tqdm is None:
        print(f"Running {total_tasks} benchmark cases", flush=True)
        for aug_name, factory in aug_items:
            results["benchmarks"][aug_name] = {}
            for cfg in BENCHMARK_CONFIGS:
                cfg_key = _config_key(cfg)
                if config_filter is not None and cfg_key not in config_filter:
                    continue

                case_idx += 1
                print(f"[{case_idx}/{total_tasks}] {aug_name} {cfg_key}", flush=True)
                images = _generate_images(cfg, seed=seed)
                ia.seed(seed)
                try:
                    aug = factory()
                except Exception as exc:
                    results["benchmarks"][aug_name][cfg_key] = {"error": f"factory_error: {exc!r}"}
                    if fail_fast:
                        raise
                    continue

                try:
                    metrics = cpu.benchmark_augmenter(
                        aug,
                        images,
                        iterations=iterations,
                        warmup=warmup,
                        progress_desc=f"{aug_name} {cfg_key}",
                    )
                except Exception as exc:
                    results["benchmarks"][aug_name][cfg_key] = {"error": repr(exc)}
                    if fail_fast:
                        raise
                else:
                    results["benchmarks"][aug_name][cfg_key] = metrics
    else:
        with tqdm(total=total_tasks, desc="Augmenters", unit="bench", leave=True, dynamic_ncols=True) as pbar:
            for aug_name, factory in aug_items:
                results["benchmarks"][aug_name] = {}
                for cfg in BENCHMARK_CONFIGS:
                    cfg_key = _config_key(cfg)
                    if config_filter is not None and cfg_key not in config_filter:
                        continue

                    pbar.set_postfix_str(f"{aug_name} {cfg_key}")
                    images = _generate_images(cfg, seed=seed)
                    ia.seed(seed)
                    try:
                        aug = factory()
                    except Exception as exc:
                        results["benchmarks"][aug_name][cfg_key] = {"error": f"factory_error: {exc!r}"}
                        pbar.update(1)
                        if fail_fast:
                            raise
                        continue

                    try:
                        metrics = cpu.benchmark_augmenter(
                            aug,
                            images,
                            iterations=iterations,
                            warmup=warmup,
                            progress_desc=f"{aug_name} {cfg_key}",
                        )
                    except Exception as exc:
                        results["benchmarks"][aug_name][cfg_key] = {"error": repr(exc)}
                        if fail_fast:
                            raise
                    else:
                        results["benchmarks"][aug_name][cfg_key] = metrics
                    pbar.update(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.date.today().isoformat()
    out_path = output_dir / f"{platform_name}_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["cpu", "apple", "nvidia"], default="cpu")
    parser.add_argument("--output", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmenters",
        type=str,
        default="",
        help="Comma-separated augmenter names; empty means all.",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated config keys (e.g. 16x256x256x3); empty means all.",
    )
    parser.add_argument("--list-augmenters", action="store_true")
    parser.add_argument("--list-configs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--label", type=str, default="")
    args = parser.parse_args()

    if args.list_augmenters:
        for name in sorted(AUGMENTERS.keys()):
            print(name)
        return

    if args.list_configs:
        for cfg in BENCHMARK_CONFIGS:
            print(_config_key(cfg))
        return

    augmenter_filter = (
        {part.strip() for part in args.augmenters.split(",") if part.strip()}
        if args.augmenters.strip()
        else None
    )
    config_filter = (
        {part.strip() for part in args.configs.split(",") if part.strip()}
        if args.configs.strip()
        else None
    )
    label = args.label.strip() or None

    run_benchmarks(
        platform_name=args.platform,
        output_dir=args.output,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        augmenter_filter=augmenter_filter,
        config_filter=config_filter,
        fail_fast=args.fail_fast,
        label=label,
    )


if __name__ == "__main__":
    main()
