# imgaug2.parameters Module

Stochastic parameters for fine-grained control over randomness.

## Import

```python
from imgaug2 import parameters as iap
```

## Base Class

### StochasticParameter

All parameters inherit from `StochasticParameter`:

```python
class StochasticParameter(metaclass=ABCMeta):
    def draw_sample(self, random_state=None):
        """Draw a single sample."""

    def draw_samples(self, size, random_state=None):
        """Draw multiple samples."""
```

## Continuous Distributions

### Uniform

Uniform distribution over [a, b].

```python
param = iap.Uniform(0, 1.0)
param = iap.Uniform(a=0, b=1.0)

value = param.draw_sample()
values = param.draw_samples(100)
```

### Normal

Normal (Gaussian) distribution.

```python
param = iap.Normal(loc=0, scale=1.0)

# With custom mean and standard deviation
param = iap.Normal(loc=10, scale=5)
```

### TruncatedNormal

Normal distribution bounded to [low, high].

```python
param = iap.TruncatedNormal(loc=0, scale=1.0, low=-2, high=2)
```

### Laplace

Laplace distribution (sharper peak than normal).

```python
param = iap.Laplace(loc=0, scale=1.0)
```

### Beta

Beta distribution (values in [0, 1]).

```python
param = iap.Beta(alpha=2, beta=5)
```

### ChiSquare

Chi-square distribution.

```python
param = iap.ChiSquare(df=4)
```

### Weibull

Weibull distribution.

```python
param = iap.Weibull(a=2)
```

## Discrete Distributions

### Deterministic

Always returns the same value.

```python
param = iap.Deterministic(5)
```

### DiscreteUniform

Integer uniform distribution over [a, b].

```python
param = iap.DiscreteUniform(0, 10)
```

### Binomial

Binomial distribution.

```python
param = iap.Binomial(n=10, p=0.5)
```

### Poisson

Poisson distribution.

```python
param = iap.Poisson(lam=5)
```

### Choice

Random choice from list.

```python
# Uniform choice
param = iap.Choice([0, 90, 180, 270])

# Weighted choice
param = iap.Choice([0, 90, 180, 270], p=[0.5, 0.2, 0.2, 0.1])

# Choice of other parameters
param = iap.Choice([
    iap.Uniform(0, 0.5),
    iap.Uniform(2.0, 3.0)
])
```

## Arithmetic Operations

### Add

Add value or parameter.

```python
param = iap.Add(iap.Uniform(0, 1), 0.5)
# Or using operator
param = iap.Uniform(0, 1) + 0.5
```

### Multiply

Multiply by value or parameter.

```python
param = iap.Multiply(iap.Normal(0, 1), 10)
# Or using operator
param = iap.Normal(0, 1) * 10
```

### Divide

Divide by value or parameter.

```python
param = iap.Divide(iap.Uniform(1, 10), 2)
# Or using operator
param = iap.Uniform(1, 10) / 2
```

### Power

Raise to power.

```python
param = iap.Power(iap.Uniform(0, 1), 2)
# Or using operator
param = iap.Uniform(0, 1) ** 2
```

### Absolute

Absolute value.

```python
param = iap.Absolute(iap.Normal(0, 10))
```

### Clip

Clip values to range.

```python
param = iap.Clip(iap.Normal(0, 10), low=-20, high=20)
```

## Sign Operations

### RandomSign

Randomly negate values.

```python
param = iap.RandomSign(iap.Uniform(0, 10))  # Values in [-10, 10]
```

### ForceSign

Force positive or negative.

```python
param = iap.ForceSign(iap.Normal(0, 10), positive=True)
```

### Positive / Negative

Convenience wrappers.

```python
param = iap.Positive(iap.Normal(0, 10))  # Force positive
param = iap.Negative(iap.Normal(0, 10))  # Force negative
```

## Usage with Augmenters

```python
import imgaug2.augmenters as iaa
from imgaug2 import parameters as iap

# Normal distribution for rotation
aug = iaa.Affine(rotate=iap.Normal(0, 15))

# Bimodal distribution for blur
aug = iaa.GaussianBlur(sigma=iap.Choice([
    iap.Uniform(0, 0.5),    # Subtle blur
    iap.Uniform(2.0, 3.0)   # Strong blur
]))

# Combined parameters
aug = iaa.Multiply(
    mul=iap.Clip(iap.Normal(1.0, 0.2), 0.5, 1.5)
)
```

## Sampling Examples

```python
from imgaug2 import parameters as iap
import numpy as np

# Create parameter
param = iap.Normal(0, 10)

# Sample single value
value = param.draw_sample()

# Sample array
values = param.draw_samples(100)

# With specific random generator
rng = np.random.default_rng(42)
value = param.draw_sample(random_state=rng)
```
