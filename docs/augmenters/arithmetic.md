# Arithmetic Augmenters

Augmenters that perform arithmetic operations on pixel values.

## Add

Add a value to all pixels.

```python
import imgaug2.augmenters as iaa

aug = iaa.Add((-40, 40))  # Add random value in [-40, 40]
aug = iaa.Add((-40, 40), per_channel=True)  # Different per channel
```

## Multiply

Multiply all pixels by a value.

```python
aug = iaa.Multiply((0.8, 1.2))  # Brightness adjustment
aug = iaa.Multiply((0.5, 1.5), per_channel=True)
```

## Dropout

Set random pixels to zero.

```python
aug = iaa.Dropout(p=(0, 0.1))  # Drop 0-10% of pixels
aug = iaa.CoarseDropout(p=(0, 0.1), size_percent=(0.02, 0.1))  # Drop rectangular regions
```

## SaltAndPepper

Add salt and pepper noise.

```python
aug = iaa.SaltAndPepper(p=(0, 0.03))  # 0-3% of pixels
aug = iaa.Salt(p=(0, 0.03))  # Only white pixels
aug = iaa.Pepper(p=(0, 0.03))  # Only black pixels
```

## AdditiveGaussianNoise

Add Gaussian noise to images.

```python
aug = iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))
aug = iaa.AdditiveGaussianNoise(scale=(0, 0.1*255), per_channel=True)
```

## AdditiveLaplaceNoise

Add Laplace noise (sharper than Gaussian).

```python
aug = iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255))
```

## AdditivePoissonNoise

Add Poisson noise (intensity-dependent).

```python
aug = iaa.AdditivePoissonNoise(lam=(0, 15))
```

## Invert

Invert pixel values (255 - pixel).

```python
aug = iaa.Invert(p=0.5)  # 50% chance to invert
aug = iaa.Invert(p=0.5, per_channel=True)
```

## Cutout

Replace rectangular regions with constant values.

```python
aug = iaa.Cutout(nb_iterations=2, size=0.2, fill_mode="constant", cval=0)
```

## All Arithmetic Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Add` | Add value to pixels |
| `AddElementwise` | Add different values to each pixel |
| `AdditiveGaussianNoise` | Gaussian noise |
| `AdditiveLaplaceNoise` | Laplace noise |
| `AdditivePoissonNoise` | Poisson noise |
| `Multiply` | Multiply pixels |
| `MultiplyElementwise` | Multiply each pixel differently |
| `Cutout` | Rectangular cutout |
| `Dropout` | Random pixel dropout |
| `CoarseDropout` | Coarse rectangular dropout |
| `Dropout2d` | Channel dropout |
| `TotalDropout` | Drop entire images |
| `ReplaceElementwise` | Replace pixels with probability |
| `ImpulseNoise` | Impulse noise |
| `SaltAndPepper` | Salt and pepper noise |
| `Salt` | Salt noise only |
| `Pepper` | Pepper noise only |
| `CoarseSaltAndPepper` | Coarse salt and pepper |
| `CoarseSalt` | Coarse salt |
| `CoarsePepper` | Coarse pepper |
| `Invert` | Invert pixel values |
| `Solarize` | Solarize effect |
| `ContrastNormalization` | Normalize contrast |
| `JpegCompression` | JPEG artifacts |
