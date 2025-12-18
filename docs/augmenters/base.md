# base (internal)

`imgaug2.augmenters.base` contains shared internal utilities and base
implementation details used by multiple augmenter modules.

Most users **should not need** this module directly — the public API is exposed
via:

```python
import imgaug2.augmenters as iaa
```

If you’re extending imgaug2 or debugging internals, this module is often the
place where common patterns (dtype checks, shared helper functions, etc.) live.

