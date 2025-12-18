# Stochastic Parameters

Stochastic parameters allow fine-grained control over randomness in augmentations.

## Basic Usage

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Simple tuple (uniform distribution)
aug = iaa.GaussianBlur(sigma=(0, 1.0))

# Explicit stochastic parameter
aug = iaa.GaussianBlur(sigma=iap.Uniform(0, 1.0))

# Normal distribution
aug = iaa.Affine(rotate=iap.Normal(0, 10))
```

## Distribution Types

### Continuous

```python
from imgaug2 import parameters as iap

iap.Uniform(0, 1.0)                              # Uniform [a, b]
iap.Normal(loc=0, scale=1.0)                     # Normal distribution
iap.TruncatedNormal(0, 1.0, low=-2, high=2)      # Bounded normal
iap.Laplace(loc=0, scale=1.0)                    # Laplace
iap.Beta(alpha=2, beta=5)                        # Beta [0, 1]
```

### Discrete

```python
from imgaug2 import parameters as iap

iap.Deterministic(5)                             # Always 5
iap.DiscreteUniform(0, 10)                       # Integer [a, b]
iap.Choice([0, 90, 180, 270])                    # Random choice
iap.Choice([0, 90], p=[0.8, 0.2])                # Weighted choice
```

## Arithmetic

```python
from imgaug2 import parameters as iap

param = iap.Uniform(0, 1) + 0.5                  # Add
param = iap.Normal(0, 1) * 10                    # Multiply
param = iap.Absolute(iap.Normal(0, 10))          # Absolute
param = iap.Clip(iap.Normal(0, 10), -20, 20)     # Clip
```

See [API Reference](api/parameters.md) for full documentation.
