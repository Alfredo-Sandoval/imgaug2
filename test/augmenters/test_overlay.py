"""Tests for deprecated overlay module.

The overlay module functions have been removed.
Use imgaug2.augmenters.blend directly instead.
"""

import unittest


class TestOverlayModuleDeprecation(unittest.TestCase):
    def test_module_exists_with_deprecation_notice(self):
        import imgaug2.augmenters.overlay as overlay

        assert "deprecated" in overlay.__doc__.lower()
