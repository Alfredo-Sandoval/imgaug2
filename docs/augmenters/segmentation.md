# Segmentation Augmenters

Augmenters that use image segmentation techniques for effects.

## Superpixels

Replace superpixels with their average color.

```python
import imgaug2.augmenters as iaa

aug = iaa.Superpixels(p_replace=(0.5, 1.0), n_segments=(50, 120))
aug = iaa.Superpixels(
    p_replace=0.5,      # Replace 50% of superpixels
    n_segments=100,     # Use ~100 segments
    max_size=128        # Downscale for speed
)
```

## Voronoi

Apply Voronoi-based segmentation effects.

```python
import imgaug2.augmenters as iaa

aug = iaa.Voronoi(
    points_sampler=iaa.UniformPointsSampler((50, 500)),
    p_replace=(0.5, 1.0)
)
```

## UniformVoronoi

Voronoi with uniformly sampled points.

```python
import imgaug2.augmenters as iaa

aug = iaa.UniformVoronoi(
    n_points=(50, 500),
    p_replace=(0.5, 1.0)
)
```

## RegularGridVoronoi

Voronoi with regular grid points (with dropout).

```python
import imgaug2.augmenters as iaa

aug = iaa.RegularGridVoronoi(
    n_rows=(10, 30),
    n_cols=(10, 30),
    p_drop_points=(0.0, 0.5)
)
```

## RelativeRegularGridVoronoi

Voronoi with relative grid size.

```python
import imgaug2.augmenters as iaa

aug = iaa.RelativeRegularGridVoronoi(
    n_rows_frac=(0.05, 0.15),
    n_cols_frac=(0.05, 0.15),
    p_drop_points=(0.0, 0.5)
)
```

## Point Samplers

For custom Voronoi configurations:

```python
import imgaug2.augmenters as iaa
from imgaug2.augmenters.segmentation import (
    UniformPointsSampler,
    RegularGridPointsSampler,
    DropoutPointsSampler
)

# Uniform random points
sampler = UniformPointsSampler(n_points=(100, 500))

# Regular grid
sampler = RegularGridPointsSampler(n_rows=10, n_cols=10)

# Grid with dropout
sampler = DropoutPointsSampler(
    RegularGridPointsSampler(n_rows=20, n_cols=20),
    p_drop=0.3
)

aug = iaa.Voronoi(points_sampler=sampler)
```

## All Segmentation Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Superpixels` | Superpixel replacement |
| `Voronoi` | Voronoi segmentation |
| `UniformVoronoi` | Voronoi with uniform points |
| `RegularGridVoronoi` | Voronoi with grid points |
| `RelativeRegularGridVoronoi` | Voronoi with relative grid |
