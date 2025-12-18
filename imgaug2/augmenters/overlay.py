"""Alias for module blend.

Deprecated module. Original name for module blend.py. Was changed in 0.2.8.

"""
from __future__ import annotations

import imgaug2.imgaug as ia
from imgaug2.augmenters import blend

_DEPRECATION_COMMENT = (
    "It has the same interface, except that the parameter "
    "`first` was renamed to `foreground` and the parameter "
    "`second` to `background`."
)


@ia.deprecated(alt_func="imgaug2.augmenters.blend.blend_alpha()",
               comment=_DEPRECATION_COMMENT)
def blend_alpha(*args, **kwargs):
    """See :func:`~imgaug2.augmenters.blend.blend_alpha`."""
    # pylint: disable=invalid-name
    return blend.blend_alpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug2.augmenters.blend.BlendAlpha",
               comment=_DEPRECATION_COMMENT)
def Alpha(*args, **kwargs):
    """See :class:`~imgaug2.augmenters.blend.BlendAlpha`."""
    # pylint: disable=invalid-name
    return blend.Alpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug2.augmenters.blend.BlendAlphaElementwise",
               comment=_DEPRECATION_COMMENT)
def AlphaElementwise(*args, **kwargs):
    """See :class:`~imgaug2.augmenters.blend.BlendAlphaElementwise`."""
    # pylint: disable=invalid-name
    return blend.AlphaElementwise(*args, **kwargs)


@ia.deprecated(alt_func="imgaug2.augmenters.blend.BlendAlphaSimplexNoise",
               comment=_DEPRECATION_COMMENT)
def SimplexNoiseAlpha(*args, **kwargs):
    """See :class:`~imgaug2.augmenters.blend.BlendAlphaSimplexNoise`."""
    # pylint: disable=invalid-name
    return blend.SimplexNoiseAlpha(*args, **kwargs)


@ia.deprecated(alt_func="imgaug2.augmenters.blend.BlendAlphaFrequencyNoise",
               comment=_DEPRECATION_COMMENT)
def FrequencyNoiseAlpha(*args, **kwargs):
    """See :class:`~imgaug2.augmenters.blend.BlendAlphaFrequencyNoise`."""
    # pylint: disable=invalid-name
    return blend.FrequencyNoiseAlpha(*args, **kwargs)
