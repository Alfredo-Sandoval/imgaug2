import sys
import imgaug2.augmenters.contrast as contrastlib

print(f"Contrast module: {contrastlib}")
print(f"Has _ContrastFuncWrapper: {hasattr(contrastlib, '_ContrastFuncWrapper')}")
try:
    print(f"_ContrastFuncWrapper: {contrastlib._ContrastFuncWrapper}")
except AttributeError as e:
    print(f"Error accessing _ContrastFuncWrapper: {e}")

import imgaug2.augmenters.pillike as pillike

print("Pillike imported successfully")
