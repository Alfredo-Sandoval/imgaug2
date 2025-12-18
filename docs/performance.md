# Performance Guide

Practical guidance for optimizing imgaug2 augmentation pipelines.

!!! note "CPU-first (today)"
    imgaug2 currently runs augmentations on the CPU.

## Quick Reference

### Fast (use freely)
`Fliplr`, `Flipud`, `Rot90`, `Add`, `Multiply`, `Dropout`, `Identity`

### Moderate (reasonable overhead)
`GaussianBlur`, `Affine`, `Resize`, `LinearContrast`, `Grayscale`

### Slow (use sparingly)
`ElasticTransformation`, `PiecewiseAffine`, `Superpixels`, `BilateralBlur`, `MotionBlur`

## Optimization Strategies

### 1) Batch processing (biggest win)

Always process images in batches:

```python
# Slow: one at a time
for img in images:
    aug(image=img)

# Fast: batch
aug(images=images)
```

Batch sizes of 16-128 typically work well.

### 2) Limit expensive augmenters (probability gates)

```python
# Apply expensive ops less often
aug = iaa.Sometimes(0.2, iaa.ElasticTransformation(alpha=50, sigma=5))
```

### 3) Use `SomeOf` / `OneOf` (don’t run everything)

```python
# Instead of applying all 5
aug = iaa.Sequential([a, b, c, d, e])

# Apply random 2-3
aug = iaa.SomeOf((2, 3), [a, b, c, d, e])
```

### 4) Downscale for heavy ops (then restore size)

Many geometric augmenters scale worse than linearly with pixel count. A common
pattern is:

1. downscale
2. run the expensive op
3. resize back to the original shape

```python
import imgaug2.augmenters as iaa

aug = iaa.KeepSizeByResize(
    iaa.Sequential(
        [
            iaa.Resize(0.5, interpolation="area"),
            iaa.PiecewiseAffine(scale=0.03),
        ]
    ),
    interpolation="linear",
)
```

## Multicore

If you are CPU-bound and want higher throughput, use multiple processes.

The recommended interface is the augmenter’s pool helper:

```python
import imgaug2.augmenters as iaa
from imgaug2.augmentables.batches import Batch

aug = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-10, 10)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
    ]
)

# `processes=-1` means "all cores except one".
with aug.pool(processes=-1, seed=0) as pool:
    batches = [Batch(images=images_batch_0), Batch(images=images_batch_1)]
    batches_aug = pool.map_batches(batches, chunksize=8)
```

### Avoid thread oversubscription (important)

If you use multiprocessing, also ensure OpenCV (and BLAS libraries) don’t spin up
many threads *per process*. Otherwise you can end up with N_processes × N_threads
and your performance can get worse.

Common mitigation:

```python
import cv2
cv2.setNumThreads(0)
```

You can also set environment variables before launching your training script:

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Dataloader Integration Patterns

### PyTorch-style (augment in workers, convert to tensor after)

General guidance:

- keep data as numpy arrays while augmenting
- convert to framework tensors after augmentation (in the worker)
- seed each worker to avoid identical augmentations across workers

See: [Reproducibility & Determinism](reproducibility.md).

## Profile Your Pipeline

```python
import time
import numpy as np
import imgaug2.augmenters as iaa

images = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
          for _ in range(100)]

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20))
])

start = time.perf_counter()
for _ in range(10):
    aug(images=images)
elapsed = time.perf_counter() - start

print(f"{1000/elapsed:.0f} images/sec")
```

## Use the benchmark tooling

If you want a more structured benchmark suite (JSON output + reports), see:

- [Benchmarks](benchmarks.md)

## Memory

High memory usage:
- `ElasticTransformation` (displacement fields)
- `PiecewiseAffine` (grid transforms)
- `Superpixels` (segmentation)

For large images or limited RAM, reduce batch size or avoid these augmenters.

## NumPy 2.x

imgaug2 supports NumPy 2.x (and late 1.x, see `numpy>=1.24,<3`). No special configuration needed.
