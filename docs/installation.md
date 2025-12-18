# Installation

## Requirements

- Python 3.10+
- NumPy >=1.24,<3
- OpenCV (installed as `opencv-python-headless<4.12` by default)
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
import imgaug2.data as data

print(ia.__version__)

# Quick test
image = data.quokka(size=(256, 256))
aug = iaa.Fliplr(0.5)
image_aug = aug(image=image)
print("imgaug2 is working!")
```

## Optional Dependencies

imgaug2 keeps some heavier dependencies optional:

```bash
# Optional JIT acceleration hooks (where supported)
pip install "imgaug2[numba]"

# Optional: required for `imgaug2.augmenters.imgcorruptlike`
pip install imagecorruptions
```
