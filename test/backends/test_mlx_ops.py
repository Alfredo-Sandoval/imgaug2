import unittest

import cv2
import numpy as np


try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False
    mx = None


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsGaussianBlur(unittest.TestCase):
    def test_matches_cv2_uint8(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        sigma = 3.0

        observed = mlx_blur.gaussian_blur(image, sigma=sigma)

        ksize = 9  # matches imgaug2.augmenters.blur._compute_gaussian_blur_ksize(3.0)
        expected = cv2.GaussianBlur(
            image,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert observed.dtype == np.uint8
        assert diff.max() <= 2

    def test_preserves_type_for_mlx_inputs(self):
        from imgaug2.mlx import blur as mlx_blur

        image = mx.ones((16, 16, 3), dtype=mx.float32)

        out = mlx_blur.gaussian_blur(image, sigma=1.0)

        assert isinstance(out, mx.array)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsEasyWins(unittest.TestCase):
    def test_fliplr_matches_numpy(self):
        from imgaug2.mlx import flip as mlx_flip

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        observed = mlx_flip.fliplr(image)
        expected = image[:, ::-1, :]

        assert isinstance(observed, np.ndarray)
        assert observed.dtype == np.uint8
        np.testing.assert_array_equal(observed, expected)

    def test_flipud_matches_numpy(self):
        from imgaug2.mlx import flip as mlx_flip

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        observed = mlx_flip.flipud(image)
        expected = image[::-1, :, :]

        np.testing.assert_array_equal(observed, expected)

    def test_rot90_matches_numpy(self):
        from imgaug2.mlx import flip as mlx_flip

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(16, 20, 3), dtype=np.uint8)

        for k in [1, 2, 3]:
            observed = mlx_flip.rot90(image, k=k)
            expected = np.rot90(image, k=k)
            np.testing.assert_array_equal(observed, expected)

    def test_invert_and_solarize_match_numpy(self):
        from imgaug2.mlx import pointwise as mlx_pointwise

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        inv = mlx_pointwise.invert(image)
        expected_inv = 255 - image
        np.testing.assert_array_equal(inv, expected_inv)

        sol = mlx_pointwise.solarize(image, threshold=128)
        expected_sol = np.where(image >= 128, 255 - image, image).astype(np.uint8)
        np.testing.assert_array_equal(sol, expected_sol)

    def test_gamma_contrast_matches_reference_uint8(self):
        from imgaug2.mlx import pointwise as mlx_pointwise

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        gamma = 1.7
        observed = mlx_pointwise.gamma_contrast(image, gamma)

        x01 = image.astype(np.float32) / 255.0
        expected = np.clip((x01**gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
        np.testing.assert_array_equal(observed, expected)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsPooling(unittest.TestCase):
    def test_avg_pool_matches_imgaug_uint8(self):
        from imgaug2.mlx import pooling as mlx_pooling
        import imgaug2.imgaug as ia

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(37, 41, 3), dtype=np.uint8)

        observed = mlx_pooling.avg_pool(image, (4, 5))
        expected = ia.avg_pool(image, (4, 5))

        assert observed.dtype == np.uint8
        assert observed.shape == expected.shape

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert diff.max() <= 2

    def test_max_pool_matches_imgaug_uint8(self):
        from imgaug2.mlx import pooling as mlx_pooling
        import imgaug2.imgaug as ia

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(33, 47, 3), dtype=np.uint8)

        observed = mlx_pooling.max_pool(image, (3, 4))
        expected = ia.max_pool(image, (3, 4))

        np.testing.assert_array_equal(observed, expected)

    def test_min_pool_matches_imgaug_uint8(self):
        from imgaug2.mlx import pooling as mlx_pooling
        import imgaug2.imgaug as ia

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(33, 47, 3), dtype=np.uint8)

        observed = mlx_pooling.min_pool(image, (3, 4))
        expected = ia.min_pool(image, (3, 4))

        np.testing.assert_array_equal(observed, expected)

    def test_preserves_type_for_mlx_inputs(self):
        from imgaug2.mlx import pooling as mlx_pooling

        image = mx.ones((16, 16, 3), dtype=mx.float32)
        out = mlx_pooling.avg_pool(image, 2, preserve_dtype=False)

        assert isinstance(out, mx.array)
        assert out.shape == (8, 8, 3)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsGeometrySanity(unittest.TestCase):
    def test_affine_identity_matches_input_uint8(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        mat = np.eye(3, dtype=np.float32)

        observed = mlx_geom.affine_transform(image, mat, order=1, mode="reflect_101")
        np.testing.assert_array_equal(observed, image)

    def test_perspective_identity_matches_input_uint8(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        mat = np.eye(3, dtype=np.float32)

        observed = mlx_geom.perspective_transform(image, mat, order=1, mode="reflect_101")
        np.testing.assert_array_equal(observed, image)
