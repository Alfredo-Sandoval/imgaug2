# Stochastic Parameters

Stochastic parameters allow fine-grained control over the randomness in augmentations. Instead of using simple tuples like `(0, 1.0)`, you can use probability distributions and arithmetic operations.

## Basic Usage

Most augmenters accept stochastic parameters:

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Simple tuple (uniform distribution)
aug = iaa.GaussianBlur(sigma=(0, 1.0))

# Explicit stochastic parameter
aug = iaa.GaussianBlur(sigma=iap.Uniform(0, 1.0))

# Normal distribution
aug = iaa.Affine(rotate=iap.Normal(0, 10))

# Truncated normal (bounded)
aug = iaa.Affine(rotate=iap.TruncatedNormal(0, 10, low=-30, high=30))
```

## Distribution Types

### Continuous Distributions

```python
from imgaug2 import parameters as iap

# Uniform distribution [a, b]
iap.Uniform(0, 1.0)

# Normal (Gaussian) distribution
iap.Normal(loc=0, scale=1.0)

# Truncated normal (bounded)
iap.TruncatedNormal(loc=0, scale=1.0, low=-2, high=2)

# Laplace distribution (sharper peak than normal)
iap.Laplace(loc=0, scale=1.0)

# Beta distribution (values in [0, 1])
iap.Beta(alpha=2, beta=5)
```

### Discrete Distributions

```python
from imgaug2 import parameters as iap

# Discrete uniform [a, b] (integers)
iap.DiscreteUniform(0, 10)

# Binomial distribution
iap.Binomial(n=10, p=0.5)

# Poisson distribution
iap.Poisson(lam=5)

# Choice from list
iap.Choice([0, 90, 180, 270])

# Choice with probabilities
iap.Choice([0, 90, 180, 270], p=[0.5, 0.2, 0.2, 0.1])
```

### Deterministic Values

```python
from imgaug2 import parameters as iap

# Always return same value
iap.Deterministic(5)
```

## Arithmetic Operations

Stochastic parameters can be combined with arithmetic:

```python
from imgaug2 import parameters as iap

# Addition
param = iap.Uniform(0, 1) + 0.5  # Shift by 0.5

# Multiplication
param = iap.Normal(0, 1) * 10  # Scale by 10

# Combining parameters
param = iap.Uniform(0, 1) * iap.Choice([1, -1])  # Random sign

# Absolute value
param = iap.Absolute(iap.Normal(0, 10))

# Clipping
param = iap.Clip(iap.Normal(0, 10), -20, 20)
```

## Practical Examples

### Rotation with Normal Distribution

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Most rotations near 0, occasional large rotations
aug = iaa.Affine(rotate=iap.Normal(0, 15))
```

### Discrete Rotation Angles

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Only 90-degree rotations
aug = iaa.Affine(rotate=iap.Choice([0, 90, 180, 270]))
```

### Bimodal Distribution

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Either small blur or large blur, not medium
param = iap.Choice([
    iap.Uniform(0, 0.5),
    iap.Uniform(2.0, 3.0)
])
aug = iaa.GaussianBlur(sigma=param)
```

### Scale with Aspect Ratio Preservation

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Same scale for x and y
scale = iap.Uniform(0.8, 1.2)
aug = iaa.Affine(scale={"x": scale, "y": scale})
```

### Per-Channel Parameters

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Different parameter per channel
aug = iaa.Add(
    value=iap.Normal(0, 10),
    per_channel=True
)
```

## Sampling Parameters

You can sample values directly from parameters:

```python
from imgaug2 import parameters as iap
import numpy as np

param = iap.Normal(0, 10)

# Sample single value
value = param.draw_sample()

# Sample multiple values
values = param.draw_samples(100)

# Sample with specific random state
rs = np.random.RandomState(42)
value = param.draw_sample(random_state=rs)
```

## All Stochastic Parameters

### Continuous

| Parameter | Description |
|-----------|-------------|
| `Uniform(a, b)` | Uniform distribution in [a, b] |
| `Normal(loc, scale)` | Normal/Gaussian distribution |
| `TruncatedNormal(loc, scale, low, high)` | Bounded normal |
| `Laplace(loc, scale)` | Laplace distribution |
| `ChiSquare(df)` | Chi-square distribution |
| `Weibull(a)` | Weibull distribution |
| `Beta(alpha, beta)` | Beta distribution in [0, 1] |
| `FromLowerResolution(param, size_px)` | Spatial variation |

### Discrete

| Parameter | Description |
|-----------|-------------|
| `Deterministic(value)` | Always same value |
| `DiscreteUniform(a, b)` | Integer uniform in [a, b] |
| `Binomial(n, p)` | Binomial distribution |
| `Poisson(lam)` | Poisson distribution |
| `Choice(values, p)` | Random choice from list |

### Arithmetic

| Parameter | Description |
|-----------|-------------|
| `Add(param, value)` | Addition |
| `Multiply(param, value)` | Multiplication |
| `Divide(param, value)` | Division |
| `Power(param, value)` | Exponentiation |
| `Absolute(param)` | Absolute value |
| `Clip(param, low, high)` | Value clipping |

### Special

| Parameter | Description |
|-----------|-------------|
| `RandomSign(param)` | Random positive/negative |
| `ForceSign(param, positive)` | Force sign |
| `Positive(param)` | Force positive |
| `Negative(param)` | Force negative |
