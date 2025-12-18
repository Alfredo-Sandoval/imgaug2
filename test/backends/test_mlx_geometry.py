import unittest
from unittest import mock

import cv2
import numpy as np


try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False
    mx = None


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxGeometryAffinePerspective(unittest.TestCase):
    def test_affine_matches_cv2_constant_bilinear(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 12.0, 1.0)

        observed = mlx_geom.affine_transform(
            image,
            m,
            order=1,
            mode="constant",
            cval=0.0,
        )
        expected = cv2.warpAffine(
            image,
            m,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert diff.max() <= 8

    def test_affine_matches_cv2_reflect_nearest(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        h, w = image.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -7.0, 1.0)

        observed = mlx_geom.affine_transform(
            image,
            m,
            order=0,
            mode="reflect",
            cval=0.0,
        )
        expected = cv2.warpAffine(
            image,
            m,
            dsize=(w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REFLECT_101,
            borderValue=0.0,
        )

        # OpenCV uses fixed-point math for warps; small disagreements can happen
        # for a handful of pixels even with nearest+reflect.
        mismatched = np.any(observed != expected, axis=-1).sum()
        assert mismatched <= 10

    def test_perspective_matches_cv2_constant_bilinear(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        h, w = image.shape[:2]
        src = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
        dst = np.array([[1, 2], [w - 3, 1], [2, h - 4], [w - 2, h - 2]], dtype=np.float32)
        mat = cv2.getPerspectiveTransform(src, dst)

        observed = mlx_geom.perspective_transform(
            image,
            mat,
            order=1,
            mode="constant",
            cval=0.0,
        )
        expected = cv2.warpPerspective(
            image,
            mat,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert diff.max() <= 8

    def test_returns_mlx_array_for_mlx_inputs(self):
        from imgaug2.mlx import geometry as mlx_geom

        image = mx.ones((32, 32, 3), dtype=mx.float32)
        m = np.eye(3, dtype=np.float32)

        out_aff = mlx_geom.affine_transform(image, m, order=1)
        out_persp = mlx_geom.perspective_transform(image, m, order=1)

        assert isinstance(out_aff, mx.array)
        assert isinstance(out_persp, mx.array)
        assert out_aff.shape == image.shape
        assert out_persp.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxGeometryElasticPiecewise(unittest.TestCase):
    def test_elastic_alpha_zero_is_identity(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out = mlx_geom.elastic_transform(image, alpha=0.0, sigma=1.0, seed=0, order=1)
        assert np.array_equal(out, image)

    def test_elastic_is_deterministic_given_seed(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out1 = mlx_geom.elastic_transform(
            image, alpha=5.0, sigma=0.8, seed=123, order=1, mode="reflect"
        )
        out2 = mlx_geom.elastic_transform(
            image, alpha=5.0, sigma=0.8, seed=123, order=1, mode="reflect"
        )
        assert np.array_equal(out1, out2)

    def test_piecewise_scale_zero_is_identity(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out = mlx_geom.piecewise_affine(image, scale=0.0, nb_rows=4, nb_cols=4, seed=0, order=1)
        assert np.array_equal(out, image)

    def test_piecewise_is_deterministic_given_seed(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(6)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out1 = mlx_geom.piecewise_affine(image, scale=0.04, nb_rows=4, nb_cols=4, seed=123, order=1)
        out2 = mlx_geom.piecewise_affine(image, scale=0.04, nb_rows=4, nb_cols=4, seed=123, order=1)
        assert np.array_equal(out1, out2)

    def test_elastic_returns_mlx_array_without_host_roundtrip(self):
        from imgaug2.mlx import geometry as mlx_geom

        image = mx.ones((32, 32, 3), dtype=mx.float32)

        with mock.patch(
            "imgaug2.mlx.geometry.to_numpy",
            side_effect=AssertionError("elastic_transform should not call to_numpy for MLX inputs"),
        ):
            out = mlx_geom.elastic_transform(
                image,
                alpha=5.0,
                sigma=0.8,
                seed=0,
                order=1,
                mode="reflect",
            )

        assert isinstance(out, mx.array)
        assert out.shape == image.shape

    def test_piecewise_returns_mlx_array_without_host_roundtrip(self):
        from imgaug2.mlx import geometry as mlx_geom

        image = mx.ones((32, 32, 3), dtype=mx.float32)

        with mock.patch(
            "imgaug2.mlx.geometry.to_numpy",
            side_effect=AssertionError("piecewise_affine should not call to_numpy for MLX inputs"),
        ):
            out = mlx_geom.piecewise_affine(
                image,
                scale=0.04,
                nb_rows=4,
                nb_cols=4,
                seed=0,
                order=1,
                mode="reflect",
            )

        assert isinstance(out, mx.array)
        assert out.shape == image.shape
