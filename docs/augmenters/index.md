# Augmenters Overview

imgaug2 provides a wide variety of image augmentation techniques.

## Categories

| Category | Description | Examples |
|----------|-------------|----------|
| [meta](meta.md) | Pipeline control | `Sequential`, `SomeOf`, `Sometimes` |
| [arithmetic](arithmetic.md) | Pixel operations | `Add`, `Multiply`, `Dropout` |
| [artistic](artistic.md) | Artistic effects | `Cartoon`, `Stylize`, `Emboss` |
| [blur](blur.md) | Blurring | `GaussianBlur`, `MotionBlur` |
| [blend](blend.md) | Blending/overlaying | `BlendAlpha*` |
| [collections](collections.md) | Collections/utilities | `Sometimes`, `WithChannels` |
| [color](color.md) | Color transforms | `Grayscale`, `AddToHue` |
| [contrast](contrast.md) | Contrast | `LinearContrast`, `CLAHE` |
| [convolutional](convolutional.md) | Convolution kernels | `Sharpen`, `Emboss`, `EdgeDetect` |
| [debug](debug.md) | Debugging tools | `AssertShape`, `AssertLambda` |
| [edges](edges.md) | Edge-based effects | `DirectedEdgeDetect`, `EdgeDetect` |
| [geometric](geometric.md) | Spatial transforms | `Affine`, `Rotate`, `ElasticTransformation` |
| [flip](flip.md) | Mirroring | `Fliplr`, `Flipud` |
| [imgcorruptlike](imgcorruptlike.md) | ImageCorruptions-style | `*` |
| [pillike](pillike.md) | PIL-style operations | `*` |
| [pooling](pooling.md) | Pooling operations | `AveragePooling`, `MaxPooling` |
| [size](size.md) | Resize/crop | `Resize`, `CropAndPad` |
| [weather](weather.md) | Weather effects | `Clouds`, `Fog`, `Snow` |
| [segmentation](segmentation.md) | Superpixels | `Superpixels`, `Voronoi` |

## Quick Example

```python
import imgaug2.data as data
import imgaug2.augmenters as iaa

image = data.quokka(size=(256, 256))

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20))
])

image_aug = aug(image=image)
```

## Cheat Sheet

If you’re trying to choose augmenters quickly, start here:
[Augmenter Index (Cheat Sheet)](augmenter_index.md).

## Building Pipelines

```python
# Apply all in order
aug = iaa.Sequential([...])

# Apply N random augmenters
aug = iaa.SomeOf((1, 3), [...])

# Apply with probability
aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=1.0))

# Apply one random augmenter
aug = iaa.OneOf([...])
```

## Compatibility Notes

Some older `imgaug` submodules exist as aliases for backwards compatibility.
For example, `imgaug2.augmenters.overlay` is a deprecated alias for
`imgaug2.augmenters.blend`.
