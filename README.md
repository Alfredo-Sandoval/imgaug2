# imgaug2

**Modern image augmentation (Python 3.10+). MIT licensed.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/imgaug2)](https://pypi.org/project/imgaug2/)
[![Tests](https://github.com/Alfredo-Sandoval/imgaug2/actions/workflows/test_master.yml/badge.svg)](https://github.com/Alfredo-Sandoval/imgaug2/actions/workflows/test_master.yml)

Image augmentation for machine learning. MIT licensed, free forever.

## Gallery (Images & Annotations)

<table>
<tr>
<th>&nbsp;</th>
<th>Image</th>
<th>Heatmaps</th>
<th>Seg. Maps</th>
<th>Keypoints</th>
<th>Bounding Boxes,<br>Polygons</th>
</tr>

<!-- Line 1: Original Input -->
<tr>
<td><em>Original Input</em></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_image.jpg?raw=true" height="83" width="124" alt="input images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_heatmap.jpg?raw=true" height="83" width="124" alt="input heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_segmap.jpg?raw=true" height="83" width="124" alt="input segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_kps.jpg?raw=true" height="83" width="124" alt="input keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/noop_bbs.jpg?raw=true" height="83" width="124" alt="input bounding boxes"></td>
</tr>

<!-- Line 2: Gauss. Noise + Contrast + Sharpen -->
<tr>
<td>Gauss. Noise<br>+&nbsp;Contrast<br>+&nbsp;Sharpen</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_image.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_heatmap.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_segmap.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_kps.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/non_geometric_bbs.jpg?raw=true" height="83" width="124" alt="non geometric augmentations, applied to bounding boxes"></td>
</tr>

<!-- Line 3: Affine -->
<tr>
<td>Affine</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_image.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_heatmap.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_segmap.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_kps.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/affine_bbs.jpg?raw=true" height="83" width="124" alt="affine augmentations, applied to bounding boxes"></td>
</tr>

<!-- Line 4: Crop + Pad -->
<tr>
<td>Crop<br>+&nbsp;Pad</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_image.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_heatmap.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_segmap.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_kps.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/cropandpad_bbs.jpg?raw=true" height="83" width="124" alt="crop and pad augmentations, applied to bounding boxes"></td>
</tr>

<!-- Line 5: Fliplr + Perspective -->
<tr>
<td>Fliplr<br>+&nbsp;Perspective</td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_image.jpg" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to images"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_heatmap.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to heatmaps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_segmap.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to segmentation maps"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_kps.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to keypoints"></td>
<td><img src="https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/small_overview/fliplr_perspective_bbs.jpg?raw=true" height="83" width="124" alt="Horizontal flip and perspective transform augmentations, applied to bounding boxes"></td>
</tr>

</table>

![64 quokkas](https://raw.githubusercontent.com/aleju/imgaug-doc/master/readme_images/examples_grid.jpg?raw=true "64 quokkas")

---

## Installation

The library requires Python 3.10+.

Install via pip:

```bash
pip install imgaug2
```

Or install the latest version directly from GitHub:

```bash
pip install git+https://github.com/Alfredo-Sandoval/imgaug2.git
```

To uninstall: `pip uninstall imgaug2`

<!-- ### Development install

```bash
git clone https://github.com/Alfredo-Sandoval/imgaug2.git
cd imgaug2
pip install -e ".[dev]"
```
-->

<!-- ### Experimental acceleration (optional)

imgaug2 runs augmentations on the CPU by default. If you want to experiment with the
in-progress acceleration backends:

```bash
# Apple Silicon (MLX)
pip install mlx

# NVIDIA (CuPy) — choose a build that matches your CUDA runtime
# e.g. pip install cupy-cuda12x
```
-->

---

## Supported Augmentations

Available augmenters organized by category:

- **Arithmetic**: Add, Multiply, Dropout, Cutout, Noise, JPEG Compression
- **Blur**: Gaussian, Motion, Bilateral, Average, Median
- **Color**: Hue, Saturation, Brightness, Temperature, Grayscale
- **Contrast**: Gamma, Sigmoid, Log, CLAHE, Histogram Equalization
- **Geometric**: Affine, Perspective, Elastic, Rotate, Scale, Crop, Pad
- **Weather**: Snow, Rain, Fog, Clouds
- **Artistic**: Cartoon effects
- **And many more...**

---

## Citation

<!--
Note: the table only lists people who have their real names (publicly)
set in their github

List of username-realname matching based on
https://github.com/aleju/imgaug/graphs/contributors ordered by commits:

wkentaro            Wada, Kentaro
Erotemic            Crall, Jon
stnk20              Tanaka, Satoshi
jgraving            Graving, Jake
creinders           Reinders, Christoph     (lastname not public on github, guessed from username)
SarthakYadav        Yadav, Sarthak
nektor211           ?
joybanerjee08       Banerjee, Joy
gaborvecsei         Vecsei, Gábor
adamwkraft          Kraft, Adam
ZhengRui            Rui, Zheng
Borda               Borovec, Jirka
vallentin           Vallentin, Christian
ss18                Zhydenko, Semen
kilsenp             Pfeiffer, Kilian
kacper1095          ?
ismaelfm            Fernández, Ismael
fmder               De Rainville, François-Michel
fchouteau           ?
chi-hung            Weng, Chi-Hung
apatsekin           ?
abnera              Ayala-Acevedo, Abner
RephaelMeudec       Meudec, Raphael
Petemir             Laporte, Matias

-->

If you use imgaug2 in your research, please cite:

```bibtex
@misc{imgaug2,
  title  = {imgaug2},
  author = {Sandoval, Alfredo},
  year   = {2025},
  url    = {https://github.com/Alfredo-Sandoval/imgaug2},
  note   = {Image augmentation library. Based on the original imgaug by Jung et al.}
}
```

**Original imgaug citation:**

```bibtex
@misc{imgaug,
  title  = {imgaug},
  author = {Jung, Alexander B. and Wada, Kentaro and Crall, Jon and Tanaka, Satoshi and Graving, Jake and
            Reinders, Christoph and Yadav, Sarthak and Banerjee, Joy and Vecsei, Gábor and Kraft, Adam and
            Rui, Zheng and Borovec, Jirka and Vallentin, Christian and Zhydenko, Semen and Pfeiffer, Kilian and
            Cook, Ben and Fernández, Ismael and De Rainville, François-Michel and Weng, Chi-Hung and
            Ayala-Acevedo, Abner and Meudec, Raphael and Laporte, Matias and others},
  year   = {2020},
  url    = {https://github.com/aleju/imgaug},
  note   = {Online; accessed 01-Feb-2020}
}
```
