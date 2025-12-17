# Migration from imgaug

imgaug2 is a drop-in replacement for the original `imgaug` library.

## Import Changes

Simply change your imports:

```python
# Before
import imgaug as ia
from imgaug import augmenters as iaa

# After
import imgaug2 as ia
from imgaug2 import augmenters as iaa
```

## Package Name

```bash
# Before
pip install imgaug

# After
pip install imgaug2
```

## What's Different

- **Python 3.9+** required
- **NumPy 1.24+** required
- Internal modernization (no `six`, no `__future__` imports)
- Bug fixes from community contributions

## What's the Same

- All augmenters work the same way
- Same API for augmentables (keypoints, bounding boxes, etc.)
- Same deterministic behavior
- Same parameter interfaces
