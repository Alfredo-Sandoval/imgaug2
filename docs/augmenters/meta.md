# Meta Augmenters

Meta augmenters control how other augmenters are applied. They don't modify images directly but orchestrate the application of child augmenters.

## Sequential

Apply augmenters in sequence.

```python
import imgaug2.augmenters as iaa

aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20))
])

# Apply in random order
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
], random_order=True)
```

## SomeOf

Apply a random subset of augmenters.

```python
import imgaug2.augmenters as iaa

# Apply 1-3 of the augmenters
aug = iaa.SomeOf((1, 3), [
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.Affine(rotate=(-20, 20)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.1*255))
])

# Apply exactly 2
aug = iaa.SomeOf(2, [...])

# Apply 0-N (any number)
aug = iaa.SomeOf((0, None), [...])
```

## OneOf

Apply exactly one randomly chosen augmenter.

```python
import imgaug2.augmenters as iaa

aug = iaa.OneOf([
    iaa.GaussianBlur(sigma=(0, 1.0)),
    iaa.AverageBlur(k=(2, 5)),
    iaa.MotionBlur(k=5)
])
```

## Sometimes

Apply an augmenter with probability p.

```python
import imgaug2.augmenters as iaa

# 50% chance to apply blur
aug = iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0)))

# With else branch
aug = iaa.Sometimes(
    0.5,
    then_list=iaa.GaussianBlur(sigma=(0, 1.0)),
    else_list=iaa.Sharpen(alpha=1.0)
)
```

## WithChannels

Apply augmenters to specific channels only.

```python
import imgaug2.augmenters as iaa

# Apply only to red channel
aug = iaa.WithChannels(0, iaa.Add((-50, 50)))

# Apply to red and green
aug = iaa.WithChannels([0, 1], iaa.Multiply((0.5, 1.5)))
```

## Identity / Noop

Do nothing (useful as placeholder).

```python
import imgaug2.augmenters as iaa

aug = iaa.Identity()
# or
aug = iaa.Noop()
```

## Lambda

Apply custom function.

```python
import imgaug2.augmenters as iaa
import numpy as np

def my_func(images, random_state, parents, hooks):
    return [image + 10 for image in images]

aug = iaa.Lambda(func_images=my_func)
```

## AssertShape

Assert image shape (for debugging).

```python
import imgaug2.augmenters as iaa

# Assert images are 64x64 with 3 channels
aug = iaa.AssertShape((None, 64, 64, 3))
```

## ChannelShuffle

Randomly shuffle color channels.

```python
import imgaug2.augmenters as iaa

aug = iaa.ChannelShuffle(0.5)  # 50% chance to shuffle
```

## All Meta Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Sequential` | Apply in sequence |
| `SomeOf` | Apply N random augmenters |
| `OneOf` | Apply one random augmenter |
| `Sometimes` | Apply with probability |
| `WithChannels` | Apply to specific channels |
| `Identity` / `Noop` | Do nothing |
| `Lambda` | Custom function |
| `AssertLambda` | Assert with function |
| `AssertShape` | Assert shape |
| `ChannelShuffle` | Shuffle channels |
| `RemoveCBAsByOutOfImageFraction` | Remove out-of-image annotations |
| `ClipCBAsToImagePlanes` | Clip annotations to image |
