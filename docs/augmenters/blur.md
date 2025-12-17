# Blur Augmenters

Augmenters that apply blurring effects to images.

## GaussianBlur

Apply Gaussian blur.

```python
import imgaug2.augmenters as iaa

aug = iaa.GaussianBlur(sigma=(0, 1.0))  # Random sigma in [0, 1]
aug = iaa.GaussianBlur(sigma=1.5)       # Fixed sigma
```

## AverageBlur

Apply average (box) blur.

```python
aug = iaa.AverageBlur(k=(2, 7))  # Random kernel size 2-7
aug = iaa.AverageBlur(k=5)       # Fixed 5x5 kernel
```

## MedianBlur

Apply median blur (good for salt-and-pepper noise).

```python
aug = iaa.MedianBlur(k=(3, 7))  # Random kernel size (must be odd)
```

## MotionBlur

Apply directional motion blur.

```python
aug = iaa.MotionBlur(k=15)                    # Random direction
aug = iaa.MotionBlur(k=15, angle=(-45, 45))   # Limited angle range
```

## BilateralBlur

Apply bilateral filter (edge-preserving blur).

```python
aug = iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
```

## MeanShiftBlur

Apply mean shift filtering.

```python
aug = iaa.MeanShiftBlur()
```

## All Blur Augmenters

| Augmenter | Description |
|-----------|-------------|
| `GaussianBlur` | Gaussian blur |
| `AverageBlur` | Box blur |
| `MedianBlur` | Median filter |
| `BilateralBlur` | Edge-preserving blur |
| `MotionBlur` | Directional motion blur |
| `MeanShiftBlur` | Mean shift filtering |
