import unittest
from unittest import mock

import numpy as np


try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False
    mx = None


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxAugmentersB1(unittest.TestCase):
    def test_gaussian_blur_keeps_device_for_mlx_inputs(self):
        import imgaug2.augmenters as iaa

        image = mx.ones((32, 32, 3), dtype=mx.float32)
        aug = iaa.GaussianBlur(1.2)

        with mock.patch(
            "imgaug2.mlx.blur.to_numpy",
            side_effect=AssertionError("GaussianBlur should not call to_numpy for MLX inputs"),
        ):
            out = aug.augment_image(image)

        assert isinstance(out, mx.array)
        assert out.shape == image.shape

    def test_affine_keeps_device_for_mlx_inputs(self):
        import imgaug2.augmenters as iaa

        image = mx.ones((32, 32, 3), dtype=mx.float32)
        aug = iaa.Affine(translate_px={"x": 3, "y": -2}, order=1, mode="edge")

        with mock.patch(
            "imgaug2.mlx.geometry.to_numpy",
            side_effect=AssertionError("Affine should not call to_numpy for MLX inputs"),
        ):
            out = aug.augment_image(image)

        assert isinstance(out, mx.array)
        assert out.shape == image.shape

    def test_perspective_keep_size_keeps_device_for_mlx_inputs(self):
        import imgaug2.augmenters as iaa

        image = mx.ones((32, 48, 3), dtype=mx.float32)
        aug = iaa.PerspectiveTransform(scale=0.05, keep_size=True)

        with mock.patch(
            "imgaug2.mlx.geometry.to_numpy",
            side_effect=AssertionError(
                "PerspectiveTransform should not call to_numpy for MLX inputs"
            ),
        ):
            out = aug.augment_image(image)

        assert isinstance(out, mx.array)
        assert out.shape == image.shape

    def test_rot90_keep_size_keeps_device_for_mlx_inputs(self):
        import imgaug2.augmenters as iaa

        image = mx.ones((16, 32, 3), dtype=mx.float32)
        aug = iaa.Rot90(k=1, keep_size=True)

        with mock.patch(
            "imgaug2.mlx.geometry.to_numpy",
            side_effect=AssertionError(
                "Rot90(keep_size=True) should not call to_numpy for MLX inputs"
            ),
        ):
            out = aug.augment_image(image)

        assert isinstance(out, mx.array)
        assert out.shape == image.shape

    def test_add_keeps_device_for_mlx_inputs(self):
        import imgaug2.augmenters as iaa

        image = mx.ones((16, 16, 3), dtype=mx.float32)
        aug = iaa.Add(5)

        with mock.patch(
            "imgaug2.mlx.pointwise.to_numpy",
            side_effect=AssertionError("Add should not call to_numpy for MLX inputs"),
        ):
            out = aug.augment_image(image)

        assert isinstance(out, mx.array)
        assert out.shape == image.shape

    def test_does_not_use_mlx_for_numpy_inputs(self):
        import imgaug2.augmenters as iaa

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        aug = iaa.GaussianBlur(1.0)

        with mock.patch(
            "imgaug2.mlx.blur.gaussian_blur",
            side_effect=AssertionError("B1: numpy inputs must not trigger MLX ops"),
        ):
            out = aug.augment_image(image)

        assert isinstance(out, np.ndarray)
        assert out.shape == image.shape
