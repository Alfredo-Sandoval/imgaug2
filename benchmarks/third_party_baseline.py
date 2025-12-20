#!/usr/bin/env python3
# ruff: noqa: ANN401
"""Third-party benchmark runner (CPU baseline).

This is intended to answer the practical question: "Are we competitive with a
common third-party augmentation stack for CPU augmentation workloads?"

Notes:
- The reference library is single-image oriented, so we loop over a batch for
  parity with the imgaug-style benchmark API.
- This benchmark is optional: it requires an extra dependency.

Run from repo root:
    python -m benchmarks.third_party_baseline --output benchmarks/results
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import json
import platform as _platform
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.config import BENCHMARK_CONFIGS
from benchmarks.platforms import cpu

_THIRD_PARTY_PACKAGE = "albumentations"


def _try_import_third_party() -> Any | None:
    try:
        return __import__(_THIRD_PARTY_PACKAGE)
    except Exception:
        return None


def third_party_available() -> bool:
    """Return True if the optional baseline dependency can be imported."""
    return _try_import_third_party() is not None


def _require_third_party() -> Any:
    third_party = _try_import_third_party()
    if third_party is None:  # pragma: no cover - exercised in envs without the dep
        raise SystemExit(
            "Third-party baseline dependency not installed.\n"
            "\n"
            "Install it via:\n"
            f"  python -m pip install {_THIRD_PARTY_PACKAGE}\n"
            "\n"
            "If you're using a conda environment with OpenCV installed via conda,\n"
            "installing imgaug2 with extras can trigger pip attempting to\n"
            "downgrade/replace OpenCV. In that case, prefer the command above.\n"
        )
    return third_party


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

    third_party = _try_import_third_party()
    if third_party is not None:
        info["third_party_version"] = str(getattr(third_party, "__version__", "unknown"))

    return info


def _config_key(cfg: tuple[int, int, int, int]) -> str:
    b, h, w, c = cfg
    return f"{b}x{h}x{w}x{c}"


def _generate_images(cfg: tuple[int, int, int, int], *, seed: int) -> np.ndarray:
    batch, h, w, c = cfg
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(batch, h, w, c), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class _ThirdPartyAug:
    transform: Any

    def __call__(self, *, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError(f"Expected images as (N,H,W,C), got {images.shape}.")
        n = int(images.shape[0])
        out = np.empty_like(images)
        for i in range(n):
            result = self.transform(image=images[i])
            out[i] = result["image"]
        return out


def _cv2_border_reflect101() -> int:
    import cv2

    return int(cv2.BORDER_REFLECT_101)


def _instantiate(transform_cls: Any, /, **kwargs: Any) -> Any:
    """Instantiate a third-party transform with version-tolerant kwargs."""
    try:
        sig = inspect.signature(transform_cls)
    except Exception:
        return transform_cls(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return transform_cls(**filtered)


def _third_party_factories() -> dict[str, Callable[[int, int], Any]]:
    """Build transform factories for the benchmark suite.

    Each factory takes (h, w) so size-dependent transforms (e.g. CoarseDropout)
    can match the imgaug2 config semantics.
    """
    A = _require_third_party()
    border_reflect101 = _cv2_border_reflect101()

    def _has(name: str) -> bool:
        return hasattr(A, name)

    def _get(name: str) -> Any:
        return getattr(A, name)

    def _no_op(_h: int, _w: int) -> Any:
        if _has("NoOp"):
            return _get("NoOp")(p=1.0)
        return A.Compose([], p=1.0)

    def _horizontal_flip(_h: int, _w: int) -> Any:
        return A.HorizontalFlip(p=1.0)

    def _vertical_flip(_h: int, _w: int) -> Any:
        return A.VerticalFlip(p=1.0)

    def _rot90(_h: int, _w: int) -> Any:
        # RandomRotate90 is the closest native transform; it's fast and matches
        # 90-degree rotations (but chooses k randomly).
        if _has("RandomRotate90"):
            return _get("RandomRotate90")(p=1.0)
        return A.Compose([], p=1.0)

    def _gaussian_blur_small(_h: int, _w: int) -> Any:
        if not _has("GaussianBlur"):
            raise RuntimeError("Third-party baseline missing GaussianBlur")
        return _get("GaussianBlur")(blur_limit=(3, 3), sigma_limit=(1.0, 1.0), p=1.0)

    def _gaussian_blur_large(_h: int, _w: int) -> Any:
        if not _has("GaussianBlur"):
            raise RuntimeError("Third-party baseline missing GaussianBlur")
        return _get("GaussianBlur")(blur_limit=(9, 9), sigma_limit=(3.0, 3.0), p=1.0)

    def _gauss_noise(_h: int, _w: int) -> Any:
        if _has("GaussNoise"):
            # Approximate scale=(0, 0.05*255) => sigma ~ 12.75 => std_range ~ 0.05.
            return _instantiate(
                _get("GaussNoise"),
                # v2+
                std_range=(0.0, 0.05),
                mean_range=(0.0, 0.0),
                # v1.x (kept for compatibility; ignored in v2)
                var_limit=(0.0, 163.0),
                mean=0.0,
                p=1.0,
            )
        raise RuntimeError("Third-party baseline missing GaussNoise")

    def _pixel_dropout(_h: int, _w: int) -> Any:
        if _has("PixelDropout"):
            return _get("PixelDropout")(dropout_prob=0.05, p=1.0)
        raise RuntimeError("Third-party baseline missing PixelDropout")

    def _coarse_dropout(h: int, w: int) -> Any:
        if not _has("CoarseDropout"):
            raise RuntimeError("Third-party baseline missing CoarseDropout")
        max_h = max(1, int(round(0.1 * float(h))))
        max_w = max(1, int(round(0.1 * float(w))))
        return _instantiate(
            _get("CoarseDropout"),
            # v2+
            num_holes_range=(1, 8),
            hole_height_range=(0.1, 0.1),
            hole_width_range=(0.1, 0.1),
            fill=0,
            # v1.x (kept for compatibility; ignored in v2)
            max_holes=8,
            max_height=max_h,
            max_width=max_w,
            min_holes=1,
            min_height=1,
            min_width=1,
            fill_value=0,
            p=1.0,
        )

    def _to_gray(_h: int, _w: int) -> Any:
        if _has("ToGray"):
            return _get("ToGray")(p=1.0)
        if _has("ToGrayV2"):
            return _get("ToGrayV2")(p=1.0)
        raise RuntimeError("Third-party baseline missing ToGray")

    def _contrast(_h: int, _w: int) -> Any:
        # Not identical to imgaug2 LinearContrast; this is a close CPU baseline.
        if _has("RandomBrightnessContrast"):
            return _get("RandomBrightnessContrast")(brightness_limit=0.0, contrast_limit=0.25, p=1.0)
        raise RuntimeError("Third-party baseline missing RandomBrightnessContrast")

    def _clahe(_h: int, _w: int) -> Any:
        if _has("CLAHE"):
            return _instantiate(_get("CLAHE"), p=1.0)
        raise RuntimeError("Third-party baseline missing CLAHE")

    def _affine_rotate(_h: int, _w: int) -> Any:
        if _has("Affine"):
            return _instantiate(
                _get("Affine"),
                rotate=(-25, 25),
                translate_percent=None,
                scale=(1.0, 1.0),
                shear=(0.0, 0.0),
                interpolation=1,  # cv2.INTER_LINEAR
                # v2+
                border_mode=border_reflect101,
                fill=0,
                # v1.x (kept for compatibility; ignored in v2)
                mode=border_reflect101,
                cval=0,
                p=1.0,
            )
        if _has("ShiftScaleRotate"):
            return _instantiate(
                _get("ShiftScaleRotate"),
                shift_limit=0.0,
                scale_limit=0.0,
                rotate_limit=25,
                interpolation=1,
                border_mode=border_reflect101,
                value=0,
                p=1.0,
            )
        raise RuntimeError("Third-party baseline missing Affine/ShiftScaleRotate")

    def _affine_scale(_h: int, _w: int) -> Any:
        if _has("Affine"):
            return _instantiate(
                _get("Affine"),
                rotate=0.0,
                translate_percent=None,
                scale=(0.8, 1.2),
                shear=(0.0, 0.0),
                interpolation=1,
                border_mode=border_reflect101,
                fill=0,
                mode=border_reflect101,
                cval=0,
                p=1.0,
            )
        if _has("ShiftScaleRotate"):
            # scale_limit is relative (e.g. 0.2 -> [0.8,1.2])
            return _instantiate(
                _get("ShiftScaleRotate"),
                shift_limit=0.0,
                scale_limit=0.2,
                rotate_limit=0,
                interpolation=1,
                border_mode=border_reflect101,
                value=0,
                p=1.0,
            )
        raise RuntimeError("Third-party baseline missing Affine/ShiftScaleRotate")

    def _affine_all(_h: int, _w: int) -> Any:
        if _has("Affine"):
            return _instantiate(
                _get("Affine"),
                rotate=(-25, 25),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                shear=(-8, 8),
                interpolation=1,
                border_mode=border_reflect101,
                fill=0,
                mode=border_reflect101,
                cval=0,
                p=1.0,
            )
        if _has("ShiftScaleRotate"):
            return _instantiate(
                _get("ShiftScaleRotate"),
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=25,
                interpolation=1,
                border_mode=border_reflect101,
                value=0,
                p=1.0,
            )
        raise RuntimeError("Third-party baseline missing Affine/ShiftScaleRotate")

    def _perspective(_h: int, _w: int) -> Any:
        if _has("Perspective"):
            return _instantiate(
                _get("Perspective"),
                scale=(0.01, 0.1),
                keep_size=True,
                border_mode=border_reflect101,
                fill=0,
                p=1.0,
            )
        raise RuntimeError("Third-party baseline missing Perspective")

    def _elastic(_h: int, _w: int) -> Any:
        if _has("ElasticTransform"):
            return _instantiate(
                _get("ElasticTransform"),
                alpha=50,
                sigma=5,
                # v1.x (kept for compatibility; ignored in v2)
                alpha_affine=0,
                border_mode=border_reflect101,
                fill=0,
                p=1.0,
            )
        raise RuntimeError("Third-party baseline missing ElasticTransform")

    def _pipeline_light(h: int, w: int) -> Any:
        return A.Compose([_horizontal_flip(h, w), _contrast(h, w)], p=1.0)

    def _pipeline_medium(h: int, w: int) -> Any:
        return A.Compose([_horizontal_flip(h, w), _gaussian_blur_small(h, w), _affine_rotate(h, w)], p=1.0)

    def _pipeline_heavy(h: int, w: int) -> Any:
        # Approximate imgaug2 SomeOf(1,3, [...]) with OneOf([...]) for simplicity.
        if _has("OneOf"):
            pick = _get("OneOf")(
                [
                    _gaussian_blur_large(h, w),
                    _gauss_noise(h, w),
                    _contrast(h, w),
                    _affine_rotate(h, w),
                ],
                p=1.0,
            )
        else:
            pick = _gaussian_blur_large(h, w)
        return A.Compose([_horizontal_flip(h, w), pick], p=1.0)

    return {
        "Identity": _no_op,
        "Fliplr": _horizontal_flip,
        "Flipud": _vertical_flip,
        "Rot90": _rot90,
        # Geometric
        "Affine_rotate": _affine_rotate,
        "Affine_scale": _affine_scale,
        "Affine_all": _affine_all,
        "PerspectiveTransform": _perspective,
        "ElasticTransformation": _elastic,
        # Blur
        "GaussianBlur_small": _gaussian_blur_small,
        "GaussianBlur_large": _gaussian_blur_large,
        # Noise
        "AdditiveGaussianNoise": _gauss_noise,
        "Dropout": _pixel_dropout,
        "CoarseDropout": _coarse_dropout,
        # Color/contrast
        "Grayscale": _to_gray,
        "LinearContrast": _contrast,
        "CLAHE": _clahe,
        # Pipelines
        "Pipeline_light": _pipeline_light,
        "Pipeline_medium": _pipeline_medium,
        "Pipeline_heavy": _pipeline_heavy,
    }


def run_third_party_baseline_benchmarks(
    *,
    output_dir: Path,
    iterations: int,
    warmup: int,
    seed: int,
    augmenter_filter: set[str] | None,
    config_filter: set[str] | None,
    fail_fast: bool,
    label: str | None,
) -> dict[str, Any]:
    _require_third_party()

    results: dict[str, Any] = {
        "system_info": _system_info(),
        "platform": "third_party",
        "label": label or "third_party_cpu",
        "params": {
            "iterations": int(iterations),
            "warmup": int(warmup),
            "seed": int(seed),
        },
        "benchmarks": {},
    }

    factories = _third_party_factories()
    for aug_name, factory in factories.items():
        if augmenter_filter is not None and aug_name not in augmenter_filter:
            continue

        results["benchmarks"][aug_name] = {}
        for cfg in BENCHMARK_CONFIGS:
            cfg_key = _config_key(cfg)
            if config_filter is not None and cfg_key not in config_filter:
                continue

            images = _generate_images(cfg, seed=seed)
            h, w = int(images.shape[1]), int(images.shape[2])

            # Keep randomness consistent-ish across runs without polluting timing.
            random.seed(seed)
            np.random.seed(seed)

            try:
                transform = factory(h, w)
                aug = _ThirdPartyAug(transform=transform)
                metrics = cpu.benchmark_augmenter(aug, images, iterations=iterations, warmup=warmup)
            except Exception as exc:
                results["benchmarks"][aug_name][cfg_key] = {"error": repr(exc)}
                if fail_fast:
                    raise
            else:
                results["benchmarks"][aug_name][cfg_key] = metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.date.today().isoformat()
    out_path = output_dir / f"third_party_{stamp}.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
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
        for name in sorted(_third_party_factories().keys()):
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

    run_third_party_baseline_benchmarks(
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
    # Keep a nice error on missing dependencies.
    sys.exit(main())
