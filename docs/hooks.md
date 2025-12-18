# Hooks

Hooks let you **intervene in augmentation runs** without rewriting your pipeline.

Common uses:

- Disable specific augmenters dynamically (e.g. skip blur for masks/heatmaps).
- Prevent propagation into child augmenters (stop a subtree in `Sequential`).
- Preprocess/postprocess inputs per-augmenter (advanced/debug use).

Hooks are part of the public API and are available from the top-level module:

```python
import imgaug2 as ia

hooks = ia.HooksImages(...)
```

## Hooks Classes

imgaug2 provides these hook types:

- `ia.HooksImages` — hooks for images
- `ia.HooksHeatmaps` — hooks for heatmaps (currently same behavior as images)
- `ia.HooksKeypoints` — hooks for keypoints (currently same behavior as images)

## The Four Hook Callbacks

All callbacks receive:

- `images`: the current input images/batch for the augmenter being executed
- `augmenter`: the augmenter instance about to run / that just ran
- `parents`: list of parent augmenters (pipeline path)
- `default`: default decision value (only for activator/propagator)

## Activator vs Propagator (Which One?)

- Use `activator` when you want to **skip a specific augmenter** (leaf nodes like
  `GaussianBlur`, `Affine`, `Add`, ...).
- Use `propagator` when you want to **disable an entire subtree** (container nodes
  like `Sequential`, `SomeOf`, `OneOf`, ...).

In practice:

- `activator=False` means “this augmenter does nothing for this call”.
- `propagator=False` means “this augmenter may still run, but it must not call children”.

### 1) `activator`

Decides whether an augmenter is executed.

```python
def activator(images, augmenter, parents, default):
    # Return False to skip this augmenter.
    # Return `default` if you don't care / want default behavior.
    return False if augmenter.name == "blur" else default
```

### 2) `propagator`

Decides whether an augmenter is allowed to call its children.

This is useful to stop traversal of a pipeline subtree.

```python
def propagator(images, augmenter, parents, default):
    if augmenter.name == "expensive_subtree":
        return False
    return default
```

### 3) `preprocessor`

Lets you modify inputs before an augmenter runs.

```python
def preprocessor(images, augmenter, parents):
    return images
```

### 4) `postprocessor`

Lets you modify outputs after an augmenter runs.

```python
def postprocessor(images, augmenter, parents):
    return images
```

## Targeting Specific Stages (Names vs Types)

Hooks are easiest when you name the stages you care about:

```python
import imgaug2.augmenters as iaa

aug = iaa.Sequential(
    [
        iaa.GaussianBlur((0.0, 1.5), name="blur"),
        iaa.Affine(rotate=(-10, 10), name="geo"),
    ]
)
```

In your hook callback you can then target by:

- `augmenter.name == "blur"` (recommended)
- `isinstance(augmenter, iaa.GaussianBlur)` (works, but can be brittle if you later swap the augmenter)
- `parents` path checks (useful for nested pipelines)

## Example: Apply Geometry to Auxiliary Image-Like Channels (Skip Photometric)

Sometimes you have an auxiliary image-like input (depth/flow/confidence map)
stored as a plain numpy array, and you want to apply **only geometric transforms**
to it (no blur/noise/color).

If you can, prefer:

- `HeatmapsOnImage` for continuous maps
- `SegmentationMapsOnImage` for discrete label maps (nearest-neighbor warps)

But if you have plain arrays, hooks are a pragmatic solution.

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

seq = iaa.Sequential(
    [
        iaa.GaussianBlur(3.0, name="blur"),
        iaa.Dropout(0.05, name="dropout"),
        iaa.Affine(translate_px=-5, name="affine"),
    ]
)

image = data.quokka(size=(128, 128))
depth = np.tile(
    np.linspace(0.0, 1.0, 128, dtype=np.float32)[np.newaxis, :, np.newaxis],
    (128, 1, 1),
)

def activator(images, augmenter, parents, default):
    return False if augmenter.name in ["blur", "dropout"] else default

seq_det = seq.to_deterministic()

image_aug = seq_det(image=image)

# Treat the auxiliary channel as an "image batch", but skip photometric stages.
depth_aug = seq_det(images=[depth], hooks=ia.HooksImages(activator=activator))[0]
```

## Example: Disable a Whole Subtree (Propagator)

If you structure your pipeline into named sub-sequences, you can disable an
entire subtree with `propagator` (this stops child execution).

This is often cleaner than listing many augmenters in an activator.

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

seq = iaa.Sequential(
    [
        iaa.Affine(rotate=(-10, 10), name="geo"),
        iaa.Sequential(
            [
                iaa.GaussianBlur((0.0, 1.5), name="blur"),
                iaa.Multiply((0.8, 1.2), name="mul"),
            ],
            name="photometric",
        ),
    ]
)

image = data.quokka(size=(128, 128))
seg = np.zeros((128, 128), dtype=np.int32)
seg[32:96, 32:96] = 1
segmap = SegmentationMapsOnImage(seg, shape=image.shape)

def skip_photometric(_data, augmenter, parents, default):
    # If we're at the photometric subtree, don't propagate into it.
    return False if augmenter.name == "photometric" else default

seq_det = seq.to_deterministic()

# Full pipeline for images:
image_aug = seq_det(image=image)

# Geometry-only for segmentation maps:
segmap_aug = seq_det(segmentation_maps=segmap, hooks=ia.HooksHeatmaps(propagator=skip_photometric))
```

## Example: Debug/Trace What Runs (Pre/Postprocessor)

You can use preprocess/postprocess hooks to log what runs (and in what context).

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

events = []

image = data.quokka(size=(128, 128))
aug = iaa.Sequential(
    [
        iaa.Affine(rotate=(-10, 10), name="affine"),
        iaa.GaussianBlur(1.0, name="blur"),
    ]
)

def preprocessor(images, augmenter, parents):
    if augmenter.name == "affine":
        events.append(
            {
                "stage": "pre",
                "augmenter": augmenter.name,
                "parents": [p.name for p in (parents or [])],
            }
        )
    return images

hooks = ia.HooksImages(preprocessor=preprocessor)
_ = aug(image=image, hooks=hooks)

print(events)
```

## Example: Skip Expensive Ops for Large Images (Dynamic Activator)

Hooks can implement “runtime policies” like skipping expensive stages based on
the current batch’s image size.

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

aug = iaa.Sequential(
    [
        iaa.ElasticTransformation(alpha=(0, 60), sigma=(4, 8), name="elastic"),
        iaa.Affine(rotate=(-10, 10), name="affine"),
    ]
)

def activator(images, augmenter, parents, default):
    # `images[0]` works for both a python list of images and an (N,H,W,C) array.
    h, w = images[0].shape[0:2]
    is_large = (h * w) >= (512 * 512)
    if is_large and augmenter.name == "elastic":
        return False
    return default

image = data.quokka(size=(512, 512))
image_aug = aug(image=image, hooks=ia.HooksImages(activator=activator))
```

## Notes / Gotchas

- Hooks are evaluated **at runtime**, so keep callbacks fast.
- Prefer naming augmenters (`name="..."`) if you plan to target specific stages.
- Hooks are for **control flow**; they don’t change how ops are implemented.
- The first callback argument is historically named `images`, but it may be *any augmentable type*
  depending on what you are augmenting (images, heatmaps, segmentation maps, etc.).
- Keep hook decisions deterministic. If your hook uses randomness (e.g. `np.random`),
  you can break reproducibility in surprising ways.
