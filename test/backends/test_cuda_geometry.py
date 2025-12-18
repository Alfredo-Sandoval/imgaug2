import unittest

import cv2
import numpy as np


try:
    import cupy as cp

    from imgaug2.cuda import _core as cuda_core

    CUDA_AVAILABLE = cuda_core.is_available()
except Exception:
    CUDA_AVAILABLE = False
    cp = None


@unittest.skipIf(not CUDA_AVAILABLE, "CUDA/CuPy not available")
class TestCudaGeometryAffineTransform(unittest.TestCase):
    def test_matches_cv2_modes_order1_uint8(self):
        from imgaug2.cuda import geometry as cuda_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        image_gpu = cp.asarray(image)

        # Forward matrix in (x,y) coordinates.
        angle = np.deg2rad(12.0)
        cos = float(np.cos(angle))
        sin = float(np.sin(angle))
        mat = np.array(
            [
                [cos, -sin, 3.2],
                [sin, cos, -2.7],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        mode_to_cv2 = {
            "constant": cv2.BORDER_CONSTANT,
            "edge": cv2.BORDER_REPLICATE,
            "symmetric": cv2.BORDER_REFLECT,
            "reflect": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
        }

        for mode, cv2_mode in mode_to_cv2.items():
            observed = cuda_geom.affine_transform(image_gpu, mat, order=1, mode=mode)
            observed_np = cp.asnumpy(observed)

            expected = cv2.warpAffine(
                image,
                mat[:2],
                dsize=(64, 64),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2_mode,
                borderValue=float(0.0),
            )

            diff = np.abs(observed_np.astype(np.int16) - expected.astype(np.int16))
            assert observed_np.dtype == np.uint8
            assert diff.max() <= 5

    def test_matches_cv2_order0_exact_uint8(self):
        from imgaug2.cuda import geometry as cuda_geom

        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        image_gpu = cp.asarray(image)

        mat = np.array(
            [
                [1.0, 0.0, 5.0],
                [0.0, 1.0, -3.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        observed = cuda_geom.affine_transform(image_gpu, mat, order=0, mode="edge")
        observed_np = cp.asnumpy(observed)

        expected = cv2.warpAffine(
            image,
            mat[:2],
            dsize=(48, 32),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_REPLICATE,
            borderValue=float(0.0),
        )
        np.testing.assert_array_equal(observed_np, expected)
