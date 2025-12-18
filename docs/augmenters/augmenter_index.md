# Augmenter Index (Cheat Sheet)

This page is the “you want to build a good pipeline quickly” view of imgaug2.

It answers:

- Which augmenter should I reach for first?
- What are the safest defaults?
- Where are the gotchas (dtype, labels, geometry)?

For deep dives, use the per-category pages listed below.

## Golden Rules

1. **Pass images + annotations together** (best sync):
   `aug(image=image, bounding_boxes=bbs, keypoints=kps, segmentation_maps=seg, ...)`
2. If you must run separate calls, use `to_deterministic()`:
   `aug_det = aug.to_deterministic()`
3. Start with `uint8` unless you have a strong reason not to.

See: [Basics](../examples/basics.md), [All Augmentables](../examples/all_augmentables.md),
[Reproducibility](../reproducibility.md), [dtype Support](../dtype_support.md).

## Quick Pipeline Templates

### Classification (general-purpose)

```python
import imgaug2.augmenters as iaa

aug = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Affine(
            rotate=(-15, 15),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            mode="edge",
        ),
        iaa.SomeOf(
            (0, 3),
            [
                iaa.GaussianBlur((0.0, 1.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
                iaa.LinearContrast((0.75, 1.25), per_channel=0.2),
                iaa.AddToHueAndSaturation((-15, 15), per_channel=0.2),
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
            ],
            random_order=True,
        ),
    ]
)
```

### Detection / segmentation (label-safe defaults)

Prefer **geometry + mild photometric**. Strong crops/warps can invalidate labels.

```python
import imgaug2.augmenters as iaa

aug = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.Affine(
            rotate=(-10, 10),
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            scale=(0.95, 1.05),
            mode="edge",
        ),
        iaa.SomeOf(
            (0, 2),
            [
                iaa.GaussianBlur((0.0, 1.0)),
                iaa.LinearContrast((0.8, 1.2), per_channel=0.2),
                iaa.Add((-20, 20), per_channel=0.2),
            ],
            random_order=True,
        ),
    ]
)
```

## Categories (What to Use First)

| Category | When to use | “Go-to” augmenters | Docs |
|----------|-------------|--------------------|------|
| meta | build pipelines / control flow | `Sequential`, `SomeOf`, `OneOf`, `Sometimes` | [meta](meta.md) |
| geometric | move pixels (and labels) | `Affine`, `ElasticTransformation`, `PiecewiseAffine` | [geometric](geometric.md) |
| flip | cheap geometry | `Fliplr`, `Flipud` | [flip](flip.md) |
| size | resize/crop/pad | `Resize`, `CropAndPad`, `PadToFixedSize`, `CropToFixedSize` | [size](size.md) |
| blur | defocus/motion/low-quality | `GaussianBlur`, `MotionBlur`, `MedianBlur` | [blur](blur.md) |
| arithmetic | brightness/noise/dropout/artifacts | `Add`, `Multiply`, `AdditiveGaussianNoise`, `CoarseDropout`, `JpegCompression` | [arithmetic](arithmetic.md) |
| color | hue/sat/temperature/quantization | `AddToHueAndSaturation`, `Grayscale`, `ChangeColorTemperature`, `Posterize` | [color](color.md) |
| contrast | intensity distribution | `LinearContrast`, `GammaContrast`, `SigmoidContrast`, `CLAHE` | [contrast](contrast.md) |
| blend | overlays / compositing | `BlendAlpha`, `BlendAlphaMask`, `BlendAlphaSimplexNoise` | [blend](blend.md) |
| convolutional | kernel effects | `Sharpen`, `Emboss`, `EdgeDetect` | [convolutional](convolutional.md) |
| edges | edge maps | `Canny` | [edges](edges.md) |
| pooling | pooling-based distortion | `AveragePooling`, `MaxPooling`, `MedianPooling` | [pooling](pooling.md) |
| weather | clouds/fog/snow/rain | `Fog`, `Clouds`, `Snowflakes`, `Rain` | [weather](weather.md) |
| segmentation | superpixels / Voronoi visuals | `Superpixels`, `Voronoi` | [segmentation](segmentation.md) |
| imgcorruptlike | standardized corruption suite | `* (severity=1..5)` | [imgcorruptlike](imgcorruptlike.md) |
| pillike | PIL-ish ops (enhance/filters) | `Enhance*`, `Filter*`, `Autocontrast` | [pillike](pillike.md) |
| collections | prebuilt strategies | `RandAugment` | [collections](collections.md) |
| debug | write debug images | `SaveDebugImageEveryNBatches` | [debug](debug.md) |

## Imports (Common Confusions)

Most augmenters:

```python
import imgaug2.augmenters as iaa
```

Special submodules:

```python
import imgaug2.augmenters.imgcorruptlike as iaa_corrupt
import imgaug2.augmenters.pillike as iaa_pil
```

Compatibility alias:

```python
# Deprecated alias kept for old code.
import imgaug2.augmenters.overlay as iaa_overlay
```

## Reference

- Full API surface (high-level): [API Reference](../api/index.md)
- Augmenter API notes: [imgaug2.augmenters](../api/augmenters.md)

