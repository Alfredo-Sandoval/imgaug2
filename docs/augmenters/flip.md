# Flip Augmenters

Augmenters that flip/mirror images.

## Fliplr (Horizontal Flip)

Flip images horizontally (left-right).

```python
import imgaug2.augmenters as iaa

aug = iaa.Fliplr(0.5)  # 50% chance to flip
aug = iaa.Fliplr(1.0)  # Always flip
```

## Flipud (Vertical Flip)

Flip images vertically (up-down).

```python
import imgaug2.augmenters as iaa

aug = iaa.Flipud(0.5)  # 50% chance to flip
aug = iaa.Flipud(1.0)  # Always flip
```

## HorizontalFlip

Alias for Fliplr.

```python
import imgaug2.augmenters as iaa

aug = iaa.HorizontalFlip(0.5)
```

## VerticalFlip

Alias for Flipud.

```python
import imgaug2.augmenters as iaa

aug = iaa.VerticalFlip(0.5)
```

## Common Usage

```python
import imgaug2.augmenters as iaa

# Combine horizontal and vertical flips
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5)
])

# This creates 4 possible orientations:
# 1. Original
# 2. Horizontal flip
# 3. Vertical flip
# 4. Both flips (180° rotation)
```

## All Flip Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Fliplr` | Horizontal flip |
| `Flipud` | Vertical flip |
| `HorizontalFlip` | Alias for Fliplr |
| `VerticalFlip` | Alias for Flipud |
