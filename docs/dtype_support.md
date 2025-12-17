# dtype Support

This page documents the data type (dtype) support across imgaug2's augmenters.

## Overview

imgaug2 supports various NumPy data types for image arrays. However, not all augmenters support all dtypes equally well.

!!! tip "Recommendation"
    **Use `uint8` for best compatibility and performance.** It's the most thoroughly tested dtype and matches most image formats.

## Supported dtypes

| dtype | Description | Value Range | Notes |
|-------|-------------|-------------|-------|
| `uint8` | Unsigned 8-bit | 0 to 255 | **Recommended** - Best tested |
| `uint16` | Unsigned 16-bit | 0 to 65,535 | Good support |
| `uint32` | Unsigned 32-bit | 0 to 4B | Limited support |
| `uint64` | Unsigned 64-bit | 0 to 18E | Discouraged |
| `int8` | Signed 8-bit | -128 to 127 | Good support |
| `int16` | Signed 16-bit | -32K to 32K | Good support |
| `int32` | Signed 32-bit | -2B to 2B | Limited support |
| `int64` | Signed 64-bit | -9E to 9E | Discouraged |
| `float16` | Half precision | ±65K | Limited support |
| `float32` | Single precision | ±3.4E38 | Good support |
| `float64` | Double precision | ±1.8E308 | Good support |
| `float128` | Extended | Very large | Discouraged |

## Support Levels

### Full Support
The augmenter handles this dtype correctly and efficiently.

### Limited Support
The augmenter works but may:
- Lose precision
- Be slower
- Produce slightly inaccurate results

### No Support
The augmenter will either:
- Raise an error
- Produce incorrect results

## Augmenter Support Matrix

### Geometric Augmenters

| Augmenter | uint8 | float32 | uint16 | int16 |
|-----------|-------|---------|--------|-------|
| `Fliplr` | Full | Full | Full | Full |
| `Flipud` | Full | Full | Full | Full |
| `Affine` | Full | Full | Full | Full |
| `Rotate` | Full | Full | Full | Full |
| `PiecewiseAffine` | Full | Full | Limited | Limited |
| `ElasticTransformation` | Full | Full | Limited | Limited |

### Blur Augmenters

| Augmenter | uint8 | float32 | uint16 | int16 |
|-----------|-------|---------|--------|-------|
| `GaussianBlur` | Full | Full | Full | Full |
| `AverageBlur` | Full | Full | Full | Full |
| `MedianBlur` | Full | Limited | Limited | No |
| `MotionBlur` | Full | Full | Limited | Limited |

### Arithmetic Augmenters

| Augmenter | uint8 | float32 | uint16 | int16 |
|-----------|-------|---------|--------|-------|
| `Add` | Full | Full | Full | Full |
| `Multiply` | Full | Full | Full | Full |
| `Dropout` | Full | Full | Full | Full |
| `SaltAndPepper` | Full | Full | Full | Full |
| `AdditiveGaussianNoise` | Full | Full | Full | Full |

### Color Augmenters

| Augmenter | uint8 | float32 | uint16 | int16 |
|-----------|-------|---------|--------|-------|
| `Grayscale` | Full | Full | Full | Full |
| `AddToHueAndSaturation` | Full | Limited | No | No |
| `ChangeColorspace` | Full | Limited | Limited | No |

## Best Practices

### 1. Use uint8 for Training

```python
import numpy as np
import imgaug2.augmenters as iaa

# Load as uint8 (standard for images)
image = np.array(PIL.Image.open("image.jpg"))  # uint8

# Augment
aug = iaa.Sequential([...])
image_aug = aug(image=image)

# Still uint8
assert image_aug.dtype == np.uint8
```

### 2. Convert for Specific Operations

```python
import numpy as np
import imgaug2.augmenters as iaa

# If you need float operations
image_float = image.astype(np.float32) / 255.0

aug = iaa.GaussianBlur(sigma=1.0)
image_float_aug = aug(image=image_float)

# Convert back if needed
image_uint8 = (image_float_aug * 255).astype(np.uint8)
```

### 3. Check dtype After Augmentation

```python
import numpy as np
import imgaug2.augmenters as iaa

image = np.zeros((100, 100, 3), dtype=np.uint8)
aug = iaa.Add((-10, 10))
image_aug = aug(image=image)

# dtype is preserved when possible
print(f"Input dtype: {image.dtype}")
print(f"Output dtype: {image_aug.dtype}")
```

### 4. Handle Precision Loss

For applications requiring high precision (e.g., segmentation maps):

```python
import numpy as np
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage

# Use int32 for segmentation maps (not float)
segmap_arr = np.zeros((256, 256), dtype=np.int32)

# imgaug2 uses nearest-neighbor interpolation for segmaps
# to preserve class labels
```

## Handling dtype Errors

If you encounter dtype-related errors:

```python
import numpy as np
import imgaug2.augmenters as iaa

# Option 1: Convert to uint8
image = image.astype(np.uint8)

# Option 2: Convert to float32
image = image.astype(np.float32)
if image.max() > 1.0:
    image = image / 255.0

# Option 3: Use augmenters that support your dtype
# Check the documentation for specific augmenter support
```

## Performance Considerations

| dtype | Memory | Speed | Notes |
|-------|--------|-------|-------|
| uint8 | Low | Fast | Optimal for most cases |
| float32 | 4x uint8 | Medium | Needed for some operations |
| float64 | 8x uint8 | Slow | Rarely needed |
| uint16 | 2x uint8 | Medium | High dynamic range images |

For best performance, keep images as `uint8` when possible and only convert when necessary for specific operations.
