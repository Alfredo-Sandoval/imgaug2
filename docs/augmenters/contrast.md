# Contrast Augmenters

Augmenters that modify image contrast.

## LinearContrast

Linear contrast adjustment.

```python
import imgaug2.augmenters as iaa

aug = iaa.LinearContrast((0.75, 1.25))
aug = iaa.LinearContrast((0.5, 2.0), per_channel=True)
```

## GammaContrast

Gamma correction.

```python
import imgaug2.augmenters as iaa

aug = iaa.GammaContrast((0.5, 2.0))
aug = iaa.GammaContrast((0.7, 1.5), per_channel=True)
```

## SigmoidContrast

Sigmoid-based contrast adjustment.

```python
import imgaug2.augmenters as iaa

aug = iaa.SigmoidContrast(gain=(3, 10), cutoff=(0.4, 0.6))
```

## LogContrast

Logarithmic contrast adjustment.

```python
import imgaug2.augmenters as iaa

aug = iaa.LogContrast(gain=(0.6, 1.4))
```

## CLAHE

Contrast Limited Adaptive Histogram Equalization.

```python
import imgaug2.augmenters as iaa

aug = iaa.CLAHE()
aug = iaa.CLAHE(clip_limit=(1, 10), tile_grid_size_px=(3, 21))
```

## AllChannelsCLAHE

Apply CLAHE to all channels independently.

```python
import imgaug2.augmenters as iaa

aug = iaa.AllChannelsCLAHE(clip_limit=(1, 10))
```

## HistogramEqualization

Standard histogram equalization.

```python
import imgaug2.augmenters as iaa

aug = iaa.HistogramEqualization()
aug = iaa.HistogramEqualization(to_colorspace="HSV")
```

## AllChannelsHistogramEqualization

Apply histogram equalization to all channels.

```python
import imgaug2.augmenters as iaa

aug = iaa.AllChannelsHistogramEqualization()
```

## All Contrast Augmenters

| Augmenter | Description |
|-----------|-------------|
| `GammaContrast` | Gamma correction |
| `SigmoidContrast` | Sigmoid contrast |
| `LogContrast` | Logarithmic contrast |
| `LinearContrast` | Linear contrast |
| `CLAHE` | Adaptive histogram eq |
| `AllChannelsCLAHE` | CLAHE on all channels |
| `HistogramEqualization` | Histogram equalization |
| `AllChannelsHistogramEqualization` | Hist eq on all channels |
| `Equalize` | PIL-style equalization |
| `Autocontrast` | PIL-style autocontrast |
