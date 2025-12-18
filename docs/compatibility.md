# Compatibility & Aliases

imgaug2 aims to be a **drop-in continuation** of `aleju/imgaug` for most code.

At the same time, a small number of names are:

- kept as **compatibility aliases** (so old code still runs),
- but are **deprecated** in favor of clearer / more explicit APIs.

This page documents what is supported, what is recommended, and what changed.

## Recommended Imports

These are the “modern” imports we recommend in new code:

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data
```

Notes:

- `ia` is kept as the common alias (`import imgaug2 as ia`) for parity with imgaug.
- Example data (e.g. quokka) lives in `imgaug2.data`.

## Package/Module Name Changes (imgaug → imgaug2)

| imgaug (old) | imgaug2 (new) |
|-------------|----------------|
| `import imgaug as ia` | `import imgaug2 as ia` |
| `import imgaug.augmenters as iaa` | `import imgaug2.augmenters as iaa` |
| `from imgaug import parameters as iap` | `from imgaug2 import parameters as iap` |

See: [Migration Guide](migration.md).

## Deprecated but Supported Aliases

### `ia.quokka(...)` and friends

The `ia.quokka(...)` helpers are still available for backwards compatibility,
but are deprecated in favor of `imgaug2.data`.

Recommended:

```python
import imgaug2.data as data
image = data.quokka(size=(256, 256))
```

Compatibility (deprecated):

```python
import imgaug2 as ia
image = ia.quokka(size=(256, 256))  # deprecated alias
```

### `overlay` → `blend`

In the original imgaug docs/code, blending lived in `augmenters.overlay`. In imgaug2,
`overlay` is a deprecated alias for `blend`.

Recommended:

```python
import imgaug2.augmenters as iaa

aug = iaa.BlendAlpha(
    factor=(0.0, 1.0),
    foreground=iaa.GaussianBlur(2.0),
)
```

Still supported (deprecated):

```python
import imgaug2.augmenters.overlay as iaa_overlay

aug = iaa_overlay.Alpha(
    (0.0, 1.0),
    first=iaa_overlay.GaussianBlur(2.0),
)
```

#### Parameter name change in blending

Some older blending APIs used:

- `first` → now `foreground`
- `second` → now `background`

If you see old code using `first=`/`second=`, switch to `foreground=`/`background=`.

## Seed / Random State Compatibility

Many augmenters accept both `seed=` and the legacy name `random_state=`. Prefer `seed=`.

```python
import imgaug2.augmenters as iaa

aug = iaa.Affine(rotate=(-10, 10), seed=0)
```

## Deterministic Mode Compatibility

Some augmenters historically had a `deterministic=True/False` init argument.
In imgaug2, the recommended pattern is:

```python
aug_det = aug.to_deterministic()
```

See: [Reproducibility & Determinism](reproducibility.md).

## Optional Compatibility Layer (`imgaug2.compat`)

imgaug2 also ships an **optional** dict-style API:

- `imgaug2.compat.Compose(...)`
- per-transform `p=...`
- `bbox_params`, `keypoint_params` helpers

See: [Migration Guide](migration.md).

## What’s Not Guaranteed

Compatibility means “old code runs”, not “everything is identical forever”.

If you need strict output parity:

- pin versions of imgaug2 + dependencies (OpenCV, scikit-image, NumPy),
- set seeds explicitly,
- keep batch shapes consistent.
