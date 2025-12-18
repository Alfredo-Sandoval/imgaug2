# Blending and Overlaying Images

imgaug2 provides powerful blending augmenters that combine two augmentation results using alpha masks. This enables localized effects, gradients, and artistic transformations.

## Basic Alpha Blending

The core formula for alpha blending is:

```
I_blend = alpha * I_foreground + (1 - alpha) * I_background
```

### BlendAlpha

Blend augmented result with original using constant alpha:

```python
import imgaug2.augmenters as iaa

# Blend sharpening effect with original (50% each)
aug = iaa.BlendAlpha(
    factor=0.5,
    foreground=iaa.Sharpen(alpha=1.0, lightness=1.0)
)

# Random blend factor
aug = iaa.BlendAlpha(
    factor=(0.0, 1.0),
    foreground=iaa.EdgeDetect(alpha=1.0)
)

# With background augmenter (instead of original image)
aug = iaa.BlendAlpha(
    factor=0.5,
    foreground=iaa.Sharpen(alpha=1.0),
    background=iaa.GaussianBlur(sigma=2.0)
)
```

## Noise-Based Blending

Create localized blending effects using noise patterns as alpha masks.

### BlendAlphaSimplexNoise

Use simplex noise for organic, cloud-like blend patterns:

```python
import imgaug2.augmenters as iaa

# Localized edge detection effect
aug = iaa.BlendAlphaSimplexNoise(
    foreground=iaa.EdgeDetect(1.0),
    background=None,  # Original image
    per_channel=False
)

# Ghost/double exposure effect
aug = iaa.BlendAlphaSimplexNoise(
    foreground=iaa.Affine(translate_percent={"x": 0.1}),
    upscale_method="linear"
)

# Per-channel noise for color effects
aug = iaa.BlendAlphaSimplexNoise(
    foreground=iaa.Multiply((0.5, 1.5)),
    per_channel=True
)
```

### BlendAlphaFrequencyNoise

Use frequency-domain noise for different pattern sizes:

```python
import imgaug2.augmenters as iaa

# Large smooth blobs (low frequency)
aug = iaa.BlendAlphaFrequencyNoise(
    foreground=iaa.Multiply(0.5),
    exponent=-4  # Negative = large patterns
)

# Small recurring patterns (high frequency)
aug = iaa.BlendAlphaFrequencyNoise(
    foreground=iaa.Multiply(0.5),
    exponent=4   # Positive = small patterns
)

# Mixed frequencies
aug = iaa.BlendAlphaFrequencyNoise(
    foreground=iaa.Add(100),
    exponent=(-4, 4)
)
```

## Gradient Blending

Apply effects with linear gradients:

```python
import imgaug2.augmenters as iaa

# Horizontal gradient (left to right)
aug = iaa.BlendAlphaHorizontalLinearGradient(
    foreground=iaa.Add(100)
)

# Vertical gradient (top to bottom)
aug = iaa.BlendAlphaVerticalLinearGradient(
    foreground=iaa.Multiply(0.5)
)
```

## Pattern-Based Blending

### Checkerboard

```python
import imgaug2.augmenters as iaa

aug = iaa.BlendAlphaCheckerboard(
    nb_rows=8,
    nb_cols=8,
    foreground=iaa.AddToHueAndSaturation((-50, 50))
)
```

### Regular Grid

```python
import imgaug2.augmenters as iaa

aug = iaa.BlendAlphaRegularGrid(
    nb_rows=(2, 8),
    nb_cols=(2, 8),
    foreground=iaa.Multiply((0.0, 0.5))
)
```

## Color-Based Blending

Apply effects only to certain colors:

```python
import imgaug2.augmenters as iaa

# Affect only some colors
aug = iaa.BlendAlphaSomeColors(
    foreground=iaa.Grayscale(1.0)
)

# Based on segmentation
aug = iaa.BlendAlphaSegMapClassIds(
    foreground=iaa.Multiply(0.5),
    class_ids=[1, 2, 3]  # Only affect these classes
)
```

## Sigmoid Sharpening

Make blend boundaries more distinct:

```python
import imgaug2.augmenters as iaa

# Sharp boundaries in noise blend
aug = iaa.BlendAlphaSimplexNoise(
    foreground=iaa.EdgeDetect(1.0),
    sigmoid=True,           # Apply sigmoid
    sigmoid_thresh=0.5      # Threshold point
)
```

## Practical Examples

### Localized Blur

```python
aug = iaa.BlendAlphaSimplexNoise(
    foreground=iaa.GaussianBlur(sigma=3.0)
)
```

### Artistic Color Shift

```python
aug = iaa.BlendAlphaFrequencyNoise(
    foreground=iaa.AddToHueAndSaturation((50, 100)),
    exponent=(-2, 2),
    per_channel=True
)
```

### Ghost/Motion Effect

```python
aug = iaa.BlendAlphaSimplexNoise(
    foreground=iaa.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
    ),
    upscale_method="cubic"
)
```

### Vignette Effect

```python
# Darken edges using gradient
aug = iaa.BlendAlphaVerticalLinearGradient(
    foreground=iaa.Sequential([
        iaa.BlendAlphaHorizontalLinearGradient(
            foreground=iaa.Multiply(0.5)
        )
    ])
)
```

## All Blend Augmenters

| Augmenter | Description |
|-----------|-------------|
| `BlendAlpha` | Constant alpha blending |
| `BlendAlphaSimplexNoise` | Simplex noise patterns |
| `BlendAlphaFrequencyNoise` | Frequency-domain noise |
| `BlendAlphaHorizontalLinearGradient` | Horizontal gradient |
| `BlendAlphaVerticalLinearGradient` | Vertical gradient |
| `BlendAlphaRegularGrid` | Grid pattern |
| `BlendAlphaCheckerboard` | Checkerboard pattern |
| `BlendAlphaSomeColors` | Color-based masking |
| `BlendAlphaSegMapClassIds` | Segmentation-based masking |
| `BlendAlphaBoundingBoxes` | Bounding box masking |
| `BlendAlphaMask` | Custom mask generator |