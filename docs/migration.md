# Migration Guide

Migrating from the original `imgaug` to `imgaug2`.

## Package Name Change

The main change is the package name:

```python
# Before (imgaug)
import imgaug as ia
import imgaug.augmenters as iaa

# After (imgaug2)
import imgaug2 as ia
import imgaug2.augmenters as iaa
```

## Installation

```bash
# Remove old package
pip uninstall imgaug

# Install new package
pip install imgaug2
```

## API Compatibility

imgaug2 aims to be compatible with imgaug 0.4.0 for most user code.

In practice, most migrations are:

1. Update imports from `imgaug` to `imgaug2`
2. Run your pipeline once and fix any deprecation warnings (usually small renames)

See also: [Compatibility & Aliases](compatibility.md).

## Optional: Dict-Based Compose API (`imgaug2.compat`)

If you prefer a dict-based API (single call, `p=` on every transform), imgaug2 provides
an **optional compatibility layer** that wraps core imgaug2 augmenters:

```python
from imgaug2 import compat as A

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(p=0.5),
    ],
    bbox_params=A.BboxParams(format="coco"),
    keypoint_params=A.KeypointParams(remove_invisible=True),
)

out = transform(image=image, bboxes=bboxes, keypoints=keypoints)
image_aug = out["image"]
bboxes_aug = out["bboxes"]
keypoints_aug = out["keypoints"]
```

## Python Version

imgaug2 requires Python 3.10+.

## NumPy Compatibility

imgaug2 supports NumPy `>=1.24,<3` (i.e. late 1.x and all 2.x releases).

Notes:

- NumPy removed legacy scalar aliases like `np.bool`, `np.int`, `np.float`, `np.complex`, ...
  in NumPy 1.24+. imgaug2 avoids these aliases. If your own code still uses them,
  switch to the builtin types (`bool`, `int`, `float`, `complex`) or to explicit
  NumPy scalar types (`np.bool_`, `np.int64`, `np.float32`, ...).
- OpenCV 4.12+ requires NumPy `>=2` on Python 3.9+. To keep NumPy 1.x support
  out of the box, imgaug2 pins `opencv-python-headless<4.12`. If you want OpenCV
  4.12+, use NumPy 2.x and install/upgrade OpenCV explicitly.

## Search and Replace

For most projects, a simple search and replace is sufficient:

```bash
# Using sed (Linux/macOS)
find . -name "*.py" -exec sed -i 's/import imgaug/import imgaug2/g' {} +
find . -name "*.py" -exec sed -i 's/from imgaug/from imgaug2/g' {} +
```

## Renamed / Deprecated Names (Common Warnings)

These are intentionally kept as aliases so old code runs, but new code should
prefer the updated names:

| Old (works, may warn) | Recommended |
|------------------------|-------------|
| `ia.quokka(...)` | `imgaug2.data.quokka(...)` |
| `imgaug2.augmenters.overlay` | `imgaug2.augmenters.blend` |
| `Alpha(..., first=..., second=...)` | `BlendAlpha(..., foreground=..., background=...)` |
| `random_state=` (init arg) | `seed=` |
| `deterministic=` (init arg) | `to_deterministic()` |

Full details: [Compatibility & Aliases](compatibility.md).

## Getting Help

If you encounter issues migrating:

1. Check the [GitHub Issues](https://github.com/Alfredo-Sandoval/imgaug2/issues)
2. Open a new issue describing your problem
