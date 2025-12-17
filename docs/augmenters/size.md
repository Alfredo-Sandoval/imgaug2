# Size Augmenters

Augmenters that resize, crop, or pad images.

## Resize

Resize images to specific size.

```python
import imgaug2.augmenters as iaa

# Fixed size
aug = iaa.Resize({"height": 128, "width": 128})

# Scale factor
aug = iaa.Resize(0.5)  # Half size
aug = iaa.Resize((0.5, 1.5))  # Random scale

# Keep aspect ratio
aug = iaa.Resize({"height": 128, "width": "keep-aspect-ratio"})
```

## CropAndPad

Crop and/or pad images.

```python
import imgaug2.augmenters as iaa

# Crop by pixels
aug = iaa.CropAndPad(px=(-10, 10))

# Crop/pad by percentage
aug = iaa.CropAndPad(percent=(-0.1, 0.1))

# Different per side
aug = iaa.CropAndPad(
    px={"top": (-10, 10), "right": (-5, 5), "bottom": (-10, 10), "left": (-5, 5)}
)
```

## Crop

Crop images only (no padding).

```python
import imgaug2.augmenters as iaa

aug = iaa.Crop(px=(0, 16))  # Crop 0-16px from each side
aug = iaa.Crop(percent=(0, 0.1))  # Crop 0-10%
```

## Pad

Pad images only (no cropping).

```python
import imgaug2.augmenters as iaa

aug = iaa.Pad(px=(0, 16))  # Pad 0-16px on each side
aug = iaa.Pad(percent=(0, 0.1), pad_mode="edge")  # Edge padding
```

## PadToFixedSize

Pad to reach a minimum size.

```python
import imgaug2.augmenters as iaa

aug = iaa.PadToFixedSize(width=256, height=256)
aug = iaa.PadToFixedSize(width=256, height=256, position="center")
```

## CropToFixedSize

Crop to reach a maximum size.

```python
import imgaug2.augmenters as iaa

aug = iaa.CropToFixedSize(width=256, height=256)
aug = iaa.CropToFixedSize(width=256, height=256, position="uniform")
```

## CenterCropToFixedSize

Center crop to specific size.

```python
import imgaug2.augmenters as iaa

aug = iaa.CenterCropToFixedSize(width=256, height=256)
```

## CropToMultiplesOf

Crop so dimensions are multiples of a value.

```python
import imgaug2.augmenters as iaa

# Crop to multiples of 32 (common for neural networks)
aug = iaa.CropToMultiplesOf(32, 32)
```

## PadToMultiplesOf

Pad so dimensions are multiples of a value.

```python
import imgaug2.augmenters as iaa

aug = iaa.PadToMultiplesOf(32, 32)
```

## KeepSizeByResize

Resize back to original size after child augmentation.

```python
import imgaug2.augmenters as iaa

aug = iaa.KeepSizeByResize(
    iaa.CropToFixedSize(128, 128)
)
```

## All Size Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Resize` | Resize images |
| `CropAndPad` | Crop and/or pad |
| `Crop` | Crop only |
| `Pad` | Pad only |
| `PadToFixedSize` | Pad to minimum size |
| `CropToFixedSize` | Crop to maximum size |
| `CenterCropToFixedSize` | Center crop |
| `CenterPadToFixedSize` | Center pad |
| `CropToMultiplesOf` | Crop to multiples |
| `PadToMultiplesOf` | Pad to multiples |
| `CropToPowersOf` | Crop to powers of N |
| `PadToPowersOf` | Pad to powers of N |
| `CropToAspectRatio` | Crop to aspect ratio |
| `PadToAspectRatio` | Pad to aspect ratio |
| `CropToSquare` | Crop to square |
| `PadToSquare` | Pad to square |
| `KeepSizeByResize` | Maintain size after aug |
