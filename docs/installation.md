# Installation

## Requirements

- Python 3.10+
- NumPy
- OpenCV (opencv-python or opencv-python-headless)
- scikit-image
- scipy
- Pillow

## pip

```bash
pip install imgaug2
```

## From Source

```bash
pip install git+https://github.com/Alfredo-Sandoval/imgaug2.git
```

## Development Installation

```bash
git clone https://github.com/Alfredo-Sandoval/imgaug2.git
cd imgaug2
pip install -e ".[dev]"
```

## Verify Installation

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa

print(ia.__version__)

# Quick test
image = ia.quokka(size=(256, 256))
aug = iaa.Fliplr(0.5)
image_aug = aug(image=image)
print("imgaug2 is working!")
```

## Optional Dependencies

For additional features:

```bash
# For saving debug images
pip install imageio

# For visualization
pip install matplotlib
```
