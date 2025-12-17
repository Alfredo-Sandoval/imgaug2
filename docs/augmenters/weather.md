# Weather Augmenters

Augmenters that simulate weather effects.

## Clouds

Add cloud-like overlay.

```python
import imgaug2.augmenters as iaa

aug = iaa.Clouds()
```

## Fog

Add fog effect.

```python
import imgaug2.augmenters as iaa

aug = iaa.Fog()
```

## Snowflakes

Add snowflake particles.

```python
import imgaug2.augmenters as iaa

aug = iaa.Snowflakes()
aug = iaa.Snowflakes(
    flake_size=(0.1, 0.4),
    speed=(0.01, 0.05)
)
```

## Rain

Add rain effect.

```python
import imgaug2.augmenters as iaa

aug = iaa.Rain()
aug = iaa.Rain(
    drop_size=(0.1, 0.2),
    speed=(0.1, 0.3)
)
```

## FastSnowyLandscape

Convert to snowy landscape appearance.

```python
import imgaug2.augmenters as iaa

aug = iaa.FastSnowyLandscape()
aug = iaa.FastSnowyLandscape(
    lightness_threshold=(100, 200),
    lightness_multiplier=(1.0, 4.0)
)
```

## Example: Weather Pipeline

```python
import imgaug2.augmenters as iaa

aug = iaa.OneOf([
    iaa.Clouds(),
    iaa.Fog(),
    iaa.Snowflakes(),
    iaa.Rain(),
])
```

## All Weather Augmenters

| Augmenter | Description |
|-----------|-------------|
| `Clouds` | Cloud overlay |
| `Fog` | Fog effect |
| `CloudLayer` | Single cloud layer |
| `Snowflakes` | Snowflake particles |
| `SnowflakesLayer` | Single snowflake layer |
| `Rain` | Rain effect |
| `RainLayer` | Single rain layer |
| `FastSnowyLandscape` | Snowy landscape |
