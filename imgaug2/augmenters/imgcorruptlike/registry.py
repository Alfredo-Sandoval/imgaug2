from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

from imgaug2.augmenters._typing import Array
from imgaug2.compat.markers import legacy

from .core import _MISSING_PACKAGE_ERROR_MSG


@legacy(version="0.4.0")
def get_corruption_names(
    subset: Literal["common", "validation", "all"] = "common",
) -> tuple[list[str], list[Callable[..., Array]]]:
    """Get a named subset of image corruption functions.

    .. note::

        This function returns the augmentation names (as strings) *and* the
        corresponding augmentation functions, while ``get_corruption_names()``
        in ``imagecorruptions`` only returns the augmentation names.


    Parameters
    ----------
    subset : {'common', 'validation', 'all'}, optional.
        Name of the subset of image corruption functions.

    Returns
    -------
    list of str
        Names of the corruption methods, e.g. "gaussian_noise".

    list of callable
        Function corresponding to the name. Is one of the
        ``apply_*()`` functions in this module. Apply e.g.
        via ``func(image, severity=2, seed=123)``.

    """
    # import imagecorruptions, note that it is an optional dependency
    try:
        # imagecorruptions sets its own warnings filter rule via
        # warnings.simplefilter(). That rule is the in effect for the whole
        # program and not just the module. So to prevent that here
        # we use catch_warnings(), which uintuitively does not by default
        # catch warnings but saves and restores the warnings filter settings.
        with warnings.catch_warnings():
            import imagecorruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG) from None

    from .blur import (
        apply_defocus_blur,
        apply_gaussian_blur,
        apply_glass_blur,
        apply_motion_blur,
        apply_zoom_blur,
    )
    from .digital import (
        apply_brightness,
        apply_contrast,
        apply_jpeg_compression,
        apply_pixelate,
        apply_saturate,
    )
    from .noise import (
        apply_gaussian_noise,
        apply_impulse_noise,
        apply_shot_noise,
        apply_speckle_noise,
    )
    from .special import apply_elastic_transform
    from .weather import apply_fog, apply_frost, apply_snow, apply_spatter

    cnames = imagecorruptions.get_corruption_names(subset)
    name_to_func = {
        "gaussian_noise": apply_gaussian_noise,
        "shot_noise": apply_shot_noise,
        "impulse_noise": apply_impulse_noise,
        "speckle_noise": apply_speckle_noise,
        "gaussian_blur": apply_gaussian_blur,
        "glass_blur": apply_glass_blur,
        "defocus_blur": apply_defocus_blur,
        "motion_blur": apply_motion_blur,
        "zoom_blur": apply_zoom_blur,
        "fog": apply_fog,
        "frost": apply_frost,
        "snow": apply_snow,
        "spatter": apply_spatter,
        "contrast": apply_contrast,
        "brightness": apply_brightness,
        "saturate": apply_saturate,
        "jpeg_compression": apply_jpeg_compression,
        "pixelate": apply_pixelate,
        "elastic_transform": apply_elastic_transform,
    }
    funcs = [name_to_func[cname] for cname in cnames]

    return cnames, funcs


# ----------------------------------------------------------------------------
# Corruption functions
# ----------------------------------------------------------------------------
# These functions could easily be created dynamically, especially templating
# the docstrings would save many lines of code. It is intentionally not done
# here for the same reasons as in case of the augmenters. See the comment
# further below at the start of the augmenter section for details.

