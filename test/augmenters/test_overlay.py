
import warnings
import unittest
from unittest import mock

import numpy as np

import imgaug2.augmenters as iaa
import imgaug2.augmenters.overlay as overlay


class Test_blend_alpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        image_fg = np.zeros((1, 1, 3), dtype=np.uint8)
        image_bg = np.copy(image_fg)
        alpha = 1

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.blend_alpha(image_fg, image_bg, alpha)

        assert len(caught_warnings) == 1
        assert (
            "imgaug2.augmenters.blend.blend_alpha"
            in str(caught_warnings[-1].message)
        )


class TestAlpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()
        factor = 1

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.Alpha(factor, children_fg)

        assert len(caught_warnings) == 2
        assert (
            "imgaug2.augmenters.blend.BlendAlpha"
            in str(caught_warnings[0].message)
        )


class TestAlphaElementwise(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()
        factor = 1

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.AlphaElementwise(factor, children_fg)

        assert len(caught_warnings) == 2
        assert (
            "imgaug2.augmenters.blend.BlendAlphaElementwise"
            in str(caught_warnings[0].message)
        )


class TestSimplexNoiseAlpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.SimplexNoiseAlpha(children_fg)

        assert len(caught_warnings) == 2
        assert (
            "imgaug2.augmenters.blend.BlendAlphaSimplexNoise"
            in str(caught_warnings[0].message)
        )


class TestFrequencyNoiseAlpha(unittest.TestCase):
    def test_warns_that_it_is_deprecated(self):
        children_fg = iaa.Identity()

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            _ = overlay.FrequencyNoiseAlpha(first=children_fg)

        assert len(caught_warnings) == 2
        assert (
            "imgaug2.augmenters.blend.BlendAlphaFrequencyNoise"
            in str(caught_warnings[0].message)
        )
