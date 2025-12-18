# Migration Guide

Migrating from the original `imgaug` to `imgaug2`.

## Package Name Change

The main change is the package name:

```python
# Before (imgaug)
import imgaug as ia
import imgaug.augmenters as iaa

# After (imgaug2)
import imgaug2 as ia
import imgaug2.augmenters as iaa
```

## Installation

```bash
# Remove old package
pip uninstall imgaug

# Install new package
pip install imgaug2
```

## API Compatibility

imgaug2 maintains full API compatibility with imgaug 0.4.0. Your existing code should work with minimal changes:

1. Update imports from `imgaug` to `imgaug2`
2. That's it!

## Python Version

imgaug2 requires Python 3.10+.

## NumPy Compatibility

imgaug2 is compatible with NumPy 1.x and 2.x. Some deprecated NumPy type aliases have been updated.

## Search and Replace

For most projects, a simple search and replace is sufficient:

```bash
# Using sed (Linux/macOS)
find . -name "*.py" -exec sed -i 's/import imgaug/import imgaug2/g' {} +
find . -name "*.py" -exec sed -i 's/from imgaug/from imgaug2/g' {} +
```

## Getting Help

If you encounter issues migrating:

1. Check the [GitHub Issues](https://github.com/Alfredo-Sandoval/imgaug2/issues)
2. Open a new issue describing your problem
