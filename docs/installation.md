# Installation

## Requirements

- Python 3.9+
- NumPy, SciPy, OpenCV, and other image processing dependencies

## Install from PyPI

```bash
pip install imgaug2
```

## Install from Source

```bash
pip install git+https://github.com/Alfredo-Sandoval/imgaug2.git
```

## Development Install

```bash
git clone https://github.com/Alfredo-Sandoval/imgaug2.git
cd imgaug2
pip install -e ".[dev]"
```

## Optional Dependencies

### Numba Acceleration

Some operations can be accelerated with Numba:

```bash
pip install numba
```

### OpenCV Variants

The default install uses `opencv-python-headless`. For GUI features:

```bash
pip install opencv-python
```
