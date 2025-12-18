# dtype Support

## Recommendation

**Use `uint8` for best compatibility.** It's the most thoroughly tested dtype.

## Supported dtypes

| dtype | Support Level |
|-------|---------------|
| `uint8` | Full (recommended) |
| `float32` | Full |
| `uint16` | Good |
| `int16` | Good |
| `float64` | Good |
| `uint32`, `int32` | Limited |
| `uint64`, `int64`, `float128` | Discouraged |

## Best Practices

```python
from PIL import Image
import numpy as np

# Keep images as uint8
image = np.array(Image.open("image.jpg").convert("RGB"))  # uint8

# Convert only when needed
image_float = image.astype(np.float32) / 255.0

# Convert back
image_uint8 = (image_float * 255).astype(np.uint8)
```

## Notes

- For most photometric augmenters (arithmetic/color/contrast), `uint8` is the least surprising dtype.
- If you train in `float32`, consider doing a single `uint8 → float32` conversion at the *edges* of your pipeline.
- Many augmenters accept `float32` images in `[0.0, 1.0]` or `[0.0, 255.0]` — check the relevant augmenter docs.

See also: [Arithmetic augmenters](augmenters/arithmetic.md), [Color augmenters](augmenters/color.md), [Contrast augmenters](augmenters/contrast.md).
