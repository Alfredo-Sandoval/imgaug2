# Meta Augmenters

Meta augmenters don’t directly “change pixels” — they control **how other augmenters run**:

- order (`Sequential`)
- stochastic selection (`SomeOf`, `OneOf`)
- probability gates (`Sometimes`)
- routing to channels (`WithChannels`)
- assertions/debug helpers (`AssertShape`, `AssertLambda`)

They’re the backbone of most real-world imgaug2 pipelines.

## Quick Start

```python
import imgaug2.augmenters as iaa

aug = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
        iaa.SomeOf(
            (0, 3),
            [
                iaa.GaussianBlur(sigma=(0.0, 1.0)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
                iaa.LinearContrast((0.75, 1.25)),
            ],
            random_order=True,
        ),
        iaa.Affine(rotate=(-15, 15)),
    ]
)
```

## Core Meta Augmenters

### `Sequential`

Runs child augmenters in order.

```python
aug = iaa.Sequential([a, b, c])
aug = iaa.Sequential([a, b, c], random_order=True)  # permutes order per batch
```

### `SomeOf` / `OneOf`

Selects a subset of child augmenters.

```python
aug = iaa.SomeOf((1, 3), [a, b, c, d])     # choose 1–3 each batch
aug = iaa.OneOf([a, b, c])                # choose exactly one
aug = iaa.SomeOf((2, 2), [a, b, c], random_order=True)  # choose 2, apply in random order
```

### `Sometimes`

Wraps a child augmenter and executes it with probability `p`.

```python
aug = iaa.Sometimes(0.2, iaa.ElasticTransformation(alpha=20, sigma=3))
```

### `WithChannels`

Applies an augmenter only to selected channels.

```python
aug = iaa.WithChannels([0, 1], iaa.Add(10))  # apply only to channels 0 and 1
```

### `Identity` / `Noop`

“Do nothing” augmenters (useful as placeholders).

```python
aug = iaa.Identity()
aug = iaa.Noop()
```

### `Lambda`, `AssertLambda`, `AssertShape`

Advanced hooks for custom logic and assertions.

These are powerful but easy to misuse (they run Python callables and can add overhead).

```python
import imgaug2.augmenters as iaa

aug = iaa.AssertShape((None, 256, 256, 3))  # (N,H,W,C), allow any N

aug = iaa.AssertLambda(
    func_images=lambda images, *_: all(img.shape[0] > 0 for img in images),
    error_message="Found an empty image.",
)
```

### `ChannelShuffle`

Randomly permutes channels (useful for RGB-like data).

```python
aug = iaa.ChannelShuffle(1.0)
```

## Determinism, Sync, and “Perfect” Pipelines

Two key rules:

1. If you need images + annotations to stay aligned, **pass them together in one call**.
2. If you need identical transforms across *separate calls* (e.g. stereo pairs), use `to_deterministic()`.

See: [Reproducibility & Determinism](../reproducibility.md) and [Hooks](../hooks.md).

## Performance Notes

Meta augmenters are generally cheap, but they can amplify overhead if you:

- build pipelines with many tiny augmenters (Python call overhead),
- use `Lambda`/assertions in hot loops,
- apply expensive augmenters too frequently.

Practical guidance: `docs/performance.md`.

## Full List (meta)

`Sequential`, `SomeOf`, `OneOf`, `Sometimes`, `WithChannels`, `Identity`, `Noop`,
`Lambda`, `AssertLambda`, `AssertShape`, `ChannelShuffle`.
