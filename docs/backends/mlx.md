# MLX Backend

GPU-accelerated image augmentation on Apple Silicon using [MLX](https://github.com/ml-explore/mlx).

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- macOS 13.3+
- Python 3.10+

## Installation

```bash
pip install "imgaug2[mlx]"
```

Or install MLX directly:

```bash
pip install mlx
```

## Quick Start

```python
import imgaug2.mlx as mlx_ops
import numpy as np

# Check availability
if mlx_ops.is_available():
    print("MLX backend ready!")

# Load image as numpy array
image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

# Convert to MLX array
img_mlx = mlx_ops.to_mlx(image)

# Apply operations
img_mlx = mlx_ops.gaussian_blur(img_mlx, sigma=1.5)
img_mlx = mlx_ops.fliplr(img_mlx)
img_mlx = mlx_ops.gamma_contrast(img_mlx, gamma=1.2)

# Convert back to numpy
result = mlx_ops.to_numpy(img_mlx)
```

## Core Functions

### Availability & Conversion

| Function            | Description                    |
| ------------------- | ------------------------------ |
| `is_available()`    | Check if MLX backend is usable |
| `require()`         | Raise error if MLX unavailable |
| `to_mlx(arr)`       | Convert numpy array to MLX     |
| `to_numpy(arr)`     | Convert MLX array to numpy     |
| `is_mlx_array(obj)` | Check if object is MLX array   |

### Low-level Exports

| Symbol                                                | Description                                                             |
| ----------------------------------------------------- | ----------------------------------------------------------------------- |
| `MLX_AVAILABLE`                                       | Whether `mlx` imported successfully in this environment                 |
| `mx`                                                  | The `mlx.core` module (useful for creating arrays / random)             |
| `ensure_float32(arr)`                                 | Ensure an MLX array is `float32` (common fast-path dtype)               |
| `restore_dtype(result, original_dtype, is_input_mlx)` | Convert MLX results back to the original dtype (used internally by ops) |

## Routing (CPU vs MLX)

`imgaug2.mlx.router` provides a _shape-aware_ helper for deciding when MLX is likely faster.

Important: This router does **not** automatically convert NumPy arrays to MLX arrays inside the
high-level augmenter pipeline. It’s intended for explicit routing at the ops level.

| Function                                        | Description                                 |
| ----------------------------------------------- | ------------------------------------------- |
| `should_use_mlx(op, batch, height, width, ...)` | Heuristic decision: True → prefer MLX       |
| `get_backend(op, batch, height, width, ...)`    | Returns `"mlx"` or `"cpu"`                  |
| `get_routing_info(op, ...)`                     | Debug/inspection: thresholds + availability |

## Operations

### Blur

| Function        | Description                           |
| --------------- | ------------------------------------- |
| `gaussian_blur` | Gaussian blur with configurable sigma |
| `average_blur`  | Box/mean blur                         |
| `median_blur`   | Median filter                         |
| `motion_blur`   | Directional motion blur               |
| `defocus_blur`  | Lens defocus effect                   |
| `zoom_blur`     | Radial zoom blur                      |
| `glass_blur`    | Frosted glass effect                  |
| `downscale`     | Downscale then upscale (pixelation)   |

### Color

| Function                    | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `grayscale`                 | Convert to grayscale                             |
| `invert`                    | Invert colors                                    |
| `solarize`                  | Solarization effect                              |
| `posterize`                 | Reduce color levels                              |
| `equalize`                  | Histogram equalization                           |
| `autocontrast`              | Auto contrast adjustment                         |
| `clahe`                     | Contrast Limited Adaptive Histogram Equalization |
| `hue_saturation_value`      | Adjust HSV channels                              |
| `color_jitter`              | Random brightness/contrast/saturation/hue        |
| `rgb_shift`                 | Shift RGB channels                               |
| `sepia`                     | Sepia tone filter                                |
| `channel_shuffle`           | Randomly reorder channels                        |
| `channel_dropout`           | Drop random channels                             |
| `fancy_pca`                 | PCA-based color augmentation                     |
| `planckian_jitter`          | Color temperature shift                          |
| `normalize` / `denormalize` | Normalize with mean/std                          |
| `to_float` / `from_float`   | Convert between uint8 and float32                |
| `rgb_to_hsv` / `hsv_to_rgb` | Convert between RGB and HSV                      |

### Geometry

| Function                | Description                  |
| ----------------------- | ---------------------------- |
| `resize`                | Resize with interpolation    |
| `affine_transform`      | Affine transformation matrix |
| `perspective_transform` | 4-point perspective warp     |
| `elastic_transform`     | Elastic deformation          |
| `piecewise_affine`      | Local affine transforms      |
| `grid_distortion`       | Grid-based distortion        |
| `optical_distortion`    | Barrel/pincushion distortion |
| `chromatic_aberration`  | RGB channel displacement     |
| `grid_sample`           | Sample using coordinate grid |

### Flip & Rotation

| Function | Description        |
| -------- | ------------------ |
| `fliplr` | Horizontal flip    |
| `flipud` | Vertical flip      |
| `rot90`  | 90-degree rotation |

### Crop & Pad

| Function              | Description                     |
| --------------------- | ------------------------------- |
| `center_crop`         | Crop from center                |
| `random_crop`         | Random position crop            |
| `random_resized_crop` | Random crop + resize            |
| `pad`                 | Pad with various modes          |
| `pad_if_needed`       | Pad only if smaller than target |

### Blend

| Function      | Description                   |
| ------------- | ----------------------------- |
| `blend_alpha` | Alpha blend foreground/background |

### Edges

| Function | Description                 |
| -------- | --------------------------- |
| `canny`  | Canny edge detector (mask) |

### Noise

| Function                  | Description               |
| ------------------------- | ------------------------- |
| `additive_gaussian_noise` | Add Gaussian noise        |
| `multiplicative_noise`    | Multiply by noise         |
| `salt_and_pepper`         | Salt and pepper noise     |
| `shot_noise`              | Poisson (shot) noise      |
| `iso_noise`               | Camera sensor noise       |
| `spatter`                 | Dirt/rain spatter         |
| `dropout`                 | Random pixel dropout      |
| `coarse_dropout`          | Block dropout             |
| `cutout`                  | Random rectangular cutout |
| `grid_dropout`            | Grid-based dropout        |
| `random_erasing`          | Random rectangle erasing  |
| `pixel_shuffle`           | Shuffle pixels in blocks  |

### Sharpening

| Function       | Description       |
| -------------- | ----------------- |
| `sharpen`      | Sharpening filter |
| `unsharp_mask` | Unsharp masking   |
| `emboss`       | Emboss effect     |

### Convolutional

| Function   | Description                     |
| ---------- | ------------------------------- |
| `convolve` | Apply a custom convolution kernel |

### Morphology

| Function                 | Description                   |
| ------------------------ | ----------------------------- |
| `erosion`                | Morphological erosion         |
| `dilation`               | Morphological dilation        |
| `opening`                | Opening (erosion + dilation)  |
| `closing`                | Closing (dilation + erosion)  |
| `morphological_gradient` | Edge detection via morphology |

### Pooling

| Function   | Description     |
| ---------- | --------------- |
| `max_pool` | Max pooling     |
| `avg_pool` | Average pooling |
| `min_pool` | Min pooling     |

### Compression

| Function           | Description                |
| ------------------ | -------------------------- |
| `jpeg_compression` | JPEG compression artifacts |

### Artistic

| Function          | Description              |
| ----------------- | ------------------------ |
| `stylize_cartoon` | Cartoon/comic stylization |

### PIL-like

| Function              | Description                      |
| --------------------- | -------------------------------- |
| `autocontrast`        | PIL-style autocontrast           |
| `equalize`            | PIL-style histogram equalization |
| `enhance_color`       | PIL ImageEnhance.Color           |
| `enhance_contrast`    | PIL ImageEnhance.Contrast        |
| `enhance_brightness`  | PIL ImageEnhance.Brightness      |
| `enhance_sharpness`   | PIL ImageEnhance.Sharpness       |
| `filter_blur`         | PIL ImageFilter.BLUR             |
| `filter_smooth`       | PIL ImageFilter.SMOOTH           |
| `filter_smooth_more`  | PIL ImageFilter.SMOOTH_MORE      |
| `filter_edge_enhance` | PIL ImageFilter.EDGE_ENHANCE     |
| `filter_edge_enhance_more` | PIL ImageFilter.EDGE_ENHANCE_MORE |
| `filter_find_edges`   | PIL ImageFilter.FIND_EDGES       |
| `filter_contour`      | PIL ImageFilter.CONTOUR          |
| `filter_emboss`       | PIL ImageFilter.EMBOSS           |
| `filter_sharpen`      | PIL ImageFilter.SHARPEN          |
| `filter_detail`       | PIL ImageFilter.DETAIL           |

### Pointwise

| Function          | Description                |
| ----------------- | -------------------------- |
| `add`             | Add constant/array         |
| `multiply`        | Multiply by constant/array |
| `gamma_contrast`  | Gamma correction           |
| `linear_contrast` | Linear contrast adjustment |

## Pipeline Utilities

| Function                               | Description                                            |
| -------------------------------------- | ------------------------------------------------------ |
| `to_device(image, dtype=...)`          | Convert to an on-device MLX array (float32 by default) |
| `to_host(image, dtype=...)`            | Convert back to NumPy (optionally restoring dtype)     |
| `chain(image, *ops, output_dtype=...)` | Run multiple ops with minimal host roundtrips          |

```python
from imgaug2.mlx import chain, to_device, to_host

# Chain multiple operations
result = chain(
    image,
    lambda x: mlx_ops.gaussian_blur(x, sigma=1.0),
    lambda x: mlx_ops.fliplr(x),
    lambda x: mlx_ops.gamma_contrast(x, gamma=1.1),
)
```

## Hybrid Operations

Some operations execute parts on CPU due to dependencies (e.g., OpenCV). These still accept/return MLX arrays but do a host roundtrip internally:

- `jpeg_compression` — Uses OpenCV JPEG codec
- `canny` — Uses OpenCV edge detector
- `blend_alpha` — Uses NumPy blending helper
- `convolve` — Uses OpenCV filter2D
- `stylize_cartoon` — Uses OpenCV segmentation + PIL-compatible logic
- `pillike.*` — Uses PIL ImageOps/ImageEnhance/ImageFilter
- `affine_transform` / `perspective_transform` — Higher-order interpolation (order > 1)
- `elastic_transform` / `piecewise_affine` — Complex coordinate mapping

For maximum throughput, minimize hybrid calls in hot paths.

## Submodules

All public functions are also organized into submodules (useful when you want to browse by topic):

| Submodule     | Contains                |
| ------------- | ----------------------- |
| `blur`        | blur operations         |
| `blend`       | alpha blending          |
| `artistic`    | artistic filters        |
| `color`       | color/intensity ops     |
| `compression` | JPEG artifacts          |
| `convolutional` | convolution kernels   |
| `crop`        | crop/pad                |
| `edges`       | edge detection          |
| `flip`        | flips/rotations         |
| `geometry`    | warps/distortions       |
| `morphology`  | erosion/dilation/etc    |
| `noise`       | noise + dropout/cutout  |
| `pipeline`    | to_device/to_host/chain |
| `pillike`     | PIL-compatible ops      |
| `pointwise`   | add/multiply/contrast   |
| `pooling`     | min/max/avg pool        |
| `router`      | routing helpers         |
| `sharpen`     | sharpen/unsharp/emboss  |

## Performance Tips

1. **Batch operations** — Process multiple images together when possible
2. **Stay on device** — Minimize `to_numpy()` calls in pipelines
3. **Avoid hybrid ops in tight loops** — JPEG compression, complex warps cause host sync
4. **Use float32** — MLX operations work best with float32; use `to_float()` / `from_float()`

## Example: Full Pipeline

```python
import numpy as np
import imgaug2.mlx as mlx_ops

def augment_batch(images: list[np.ndarray]) -> list[np.ndarray]:
    """GPU-accelerated augmentation pipeline."""
    mlx_ops.require()  # Ensure MLX is available

    results = []
    for img in images:
        # To device
        x = mlx_ops.to_mlx(img)
        x = mlx_ops.to_float(x)

        # Augmentations
        x = mlx_ops.color_jitter(x, brightness=0.2, contrast=0.2)
        x = mlx_ops.gaussian_blur(x, sigma=np.random.uniform(0, 1.5))

        if np.random.random() > 0.5:
            x = mlx_ops.fliplr(x)

        x = mlx_ops.additive_gaussian_noise(x, scale=0.02)

        # Back to host
        x = mlx_ops.from_float(x)
        results.append(mlx_ops.to_numpy(x))

    return results
```

## Automatic Backend Selection

The router module can automatically select MLX when beneficial:

```python
from imgaug2.mlx import should_use_mlx, get_backend

# Check if MLX should be used for this operation
if should_use_mlx(image, operation="gaussian_blur"):
    # Use MLX path
    pass

# Get recommended backend info
info = get_routing_info(image)
print(info)  # {'backend': 'mlx', 'reason': 'apple_silicon_detected'}
```

## Fast Path Coverage

The following augmenters have optimized MLX implementations ("fast-paths"). When these conditions are met, the pipeline avoids converting back to NumPy/CPU, resulting in significant speedups.

| Augmenter                  | Fast-Path Status | Notes                                        |
| -------------------------- | ---------------- | -------------------------------------------- |
| `Sequential`               | ✅ Full Support  | Dispatches children efficiently              |
| `SomeOf`, `OneOf`          | ✅ Full Support  | Dispatches children efficiently              |
| `Sometimes`                | ✅ Full Support  |                                              |
| `Fliplr`, `Flipud`         | ✅ Full Support  |                                              |
| `Rot90`                    | ✅ Full Support  |                                              |
| `Multiply`, `Add`          | ✅ Full Support  | Supports broadcasting                        |
| `GammaContrast`            | ✅ Full Support  |                                              |
| `SigmoidContrast`          | ✅ Full Support  |                                              |
| `LogContrast`              | ✅ Full Support  |                                              |
| `LinearContrast`           | ✅ Full Support  |                                              |
| `Grayscale`                | ✅ Full Support  |                                              |
| `MultiplyHueAndSaturation` | ✅ Full Support  | Includes `MultiplyHue`, `MultiplySaturation` |
| `AddToHueAndSaturation`    | ✅ Full Support  | Includes `AddToHue`, `AddToSaturation`       |
| `GaussianBlur`             | ✅ Full Support  | Uses `mlx.core.fast.gaussian_blur`           |
| `AverageBlur`              | ✅ Full Support  |                                              |
| `MedianBlur`               | ⚠️ Partial       | Fallback to CPU for some kernel sizes        |
| `Resize`                   | ✅ Full Support  | Linear/Nearest interpolation                 |
| `Crop`, `Pad`              | ✅ Full Support  |                                              |
