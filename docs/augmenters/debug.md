# Debug Augmenters

Debug augmenters help you **inspect what your pipeline is doing** by generating
visualizations of batches and writing them to disk.

They are most useful when you’re building a pipeline that augments annotations
(bbs/kps/polys/segmaps/heatmaps) and want a quick sanity check that everything
stays aligned.

## SaveDebugImageEveryNBatches

Save augmented images to disk periodically.

```python
import imgaug2.augmenters as iaa

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.SaveDebugImageEveryNBatches(
        destination="debug_images/",
        interval=10,
    )
])
```

## Recommended Usage

```python
import imgaug2.augmenters as iaa
import os

# Create debug directory
os.makedirs("debug_images", exist_ok=True)

# Add debug augmenter to pipeline
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-25, 25)),
    iaa.SaveDebugImageEveryNBatches("debug_images/", interval=5),
])

# During training, images will be saved every 5 batches
for batch in dataloader:
    images_aug = aug(images=batch)
    # Images saved to debug_images/batch_000000.png, etc.
```

## Notes / Gotchas

### Destination must exist

`SaveDebugImageEveryNBatches` expects `destination` to be an existing directory.
Create it before you start training (see `os.makedirs(..., exist_ok=True)` above).

### Output files

When `destination` is a folder path, imgaug2 writes:

- `batch_000000.png`, `batch_000001.png`, … (6-digit batch counter)
- `batch_latest.png` (overwritten each time a debug image is saved)

### Batch counting is “as seen”

The internal counter increments only for batches that the augmenter sees.
If you run the debug augmenter conditionally, or re-instantiate your pipeline,
the numbering can be surprising.

### Multiprocessing / multi-worker dataloaders

If multiple processes write to the **same** folder, filenames will collide.
Use a per-worker destination directory (e.g. `debug_images/worker_0/`, …).

## All Debug Augmenters

| Augmenter | Description |
|-----------|-------------|
| `SaveDebugImageEveryNBatches` | Save images periodically |
