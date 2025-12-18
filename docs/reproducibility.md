# Reproducibility & Determinism

imgaug2 gives you two related (but different) tools:

- **Reproducibility**: re-running the same script produces the same results.
- **Determinism**: applying the *same sampled transform* to multiple inputs (e.g. image + mask, or stereo pairs).

This page documents the recommended patterns for “perfectly repeatable” pipelines.

## TL;DR (Recommended Patterns)

### 1) Reproducible run (same script → same outputs)

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa

ia.seed(42)

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20)),
])
```

### 2) Keep images + annotations in sync (best option)

Pass them **together in one call**:

```python
image_aug, segmap_aug = aug(image=image, segmentation_maps=segmap)
```

### 3) Apply identical transforms to two separate batches (stereo, teacher/student, etc.)

Use deterministic mode:

```python
aug_det = aug.to_deterministic()

left_aug = aug_det(images=left_images)
right_aug = aug_det(images=right_images)
```

This repeats the same sampled parameters **as long as batch sizes and shapes match**.

## Reproducibility vs Determinism (What They Mean)

### Reproducibility

If you run a script twice with the same seed and the same inputs, you want the same outputs.

This is usually achieved with:

- `ia.seed(<int>)` (global RNG seeding)
- keeping the pipeline structure constant
- keeping batch sizes + image shapes constant (important!)

### Determinism

If you have multiple inputs that must undergo identical transformations (e.g. image and its bounding boxes), you need to ensure they share the same sampled parameters.

imgaug2 supports that in two ways:

1. **Single-call augmentation**: pass everything together (recommended).
2. **Deterministic augmenters**: `aug_det = aug.to_deterministic()` and apply it to multiple batches.

## Seeding (Global vs Local)

### Global seeding (most common)

```python
import imgaug2 as ia
ia.seed(42)
```

This sets the seed of imgaug2’s **global RNG**, which is the default RNG for many augmenters.

!!! note
    `ia.seed()` is kept for backwards compatibility with imgaug. Internally, the preferred
    newer API is `imgaug2.random.seed(...)`, but `ia.seed(...)` is fine for end users.

### Augmenter-local seeding (advanced)

Some situations (e.g. multiprocessing/data-loader workers) benefit from explicitly seeding an augmenter:

```python
aug.seed_(entropy=123)
```

This seeds the augmenter and its children. It’s most useful when the augmenter is used across processes.

## Multiprocessing / Multi-Worker Gotchas

If you use multiprocessing (e.g. a dataloader with multiple workers), be aware:

- On `fork`-based start methods, worker processes may inherit identical RNG state.
- If each worker inherits the same state and you don’t reseed, you can end up with
  **identical augmentations across workers** (quietly reducing data diversity).

The fix is simple: **seed per worker/process**.

### Option A: imgaug2 multicore pool

For pure imgaug2 pipelines, using the built-in multicore pool is the simplest:

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

# The pool seeds child processes based on `seed=...`.
with aug.pool(processes=-1, seed=0) as pool:
    batches = [Batch(images=images_batch_0), Batch(images=images_batch_1)]
    batches_aug = pool.map_batches(batches, chunksize=8)
```

### Option B: seed inside each dataloader worker (PyTorch-style pattern)

If you use a dataloader with N workers, seed once per worker:

```python
# Example pattern (PyTorch-style) — seed each worker process.
import imgaug2 as ia
import imgaug2.augmenters as iaa

BASE_SEED = 123

def make_augmenter_for_worker(worker_id: int) -> iaa.Augmenter:
    # Ensure each worker has a different seed, but the run is still reproducible.
    worker_seed = BASE_SEED + worker_id
    ia.seed(worker_seed)
    return iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-10, 10)),
        ]
    )
```

If you keep the augmenter instance around in the worker, prefer calling `aug.seed_(entropy=...)`
once when the worker starts.

## Deterministic Augmenters (`to_deterministic()`)

### What deterministic mode guarantees

For a deterministic augmenter:

- Each augmentation call starts sampling from the **same RNG state**.
- If batch size and shapes stay the same, you get the **same sampled parameters** each time.

This is ideal for:

- stereo images (left/right)
- teacher/student pipelines
- applying a pipeline separately to images and labels while keeping transforms identical

### What deterministic mode does NOT guarantee

!!! warning
    Deterministic mode is sensitive to *batch structure*.
    If you change batch size or image shapes, random sampling will consume a different sequence of values and results will differ.

### Example: image + bounding boxes, deterministic (separate calls)

Single-call is simplest, but deterministic mode also supports separate calls:

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

image = data.quokka(size=(256, 256))
bbs = ia.BoundingBoxesOnImage(
    [ia.BoundingBox(x1=30, y1=40, x2=180, y2=200, label="obj")],
    shape=image.shape,
)

aug = iaa.Affine(rotate=(-25, 25), translate_px={"x": (-10, 10), "y": (-10, 10)})
aug_det = aug.to_deterministic()

image_aug = aug_det(image=image)
bbs_aug = aug_det(bounding_boxes=bbs)
```

## Best Practices (Perfect Pipelines)

- **Prefer single-call augmentation** for images + augmentables:
  - `aug(image=image, bounding_boxes=bbs, keypoints=kps, segmentation_maps=segmap, ...)`
- **Name important stages** (`name="..."`) so you can debug/hook them.
- **Keep batch sizes constant** when you care about exact reproducibility.
- In multiprocessing: seed per worker (or use imgaug2’s pool helpers).

## Related Docs

- Hooks: [Hooks](hooks.md)
- Benchmarks & performance: [Performance](performance.md), [Benchmarks](benchmarks.md)
- Stochastic parameters: [Stochastic Parameters](stochastic_parameters.md)
