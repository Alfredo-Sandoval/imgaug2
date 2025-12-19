import unittest

import numpy as np

try:
    import cupy as cp

    from imgaug2.cuda import _core as cuda_core

    CUDA_AVAILABLE = cuda_core.is_available()
except Exception:
    CUDA_AVAILABLE = False
    cp = None


@unittest.skipIf(not CUDA_AVAILABLE, "CUDA/CuPy not available")
class TestCudaOpsEasyWins(unittest.TestCase):
    def test_flip_shapes(self):
        from imgaug2.cuda import flip as cuda_flip

        x = cp.asarray(np.random.randint(0, 256, size=(16, 20, 3), dtype=np.uint8))

        assert cuda_flip.fliplr(x).shape == x.shape
        assert cuda_flip.flipud(x).shape == x.shape
        assert cuda_flip.rot90(x, k=1).shape == (20, 16, 3)

    def test_pointwise_preserves_dtype_uint8(self):
        from imgaug2.cuda import pointwise as cuda_pointwise

        x = cp.asarray(np.random.randint(0, 256, size=(16, 20, 3), dtype=np.uint8))
        y = cuda_pointwise.invert(x)
        assert y.dtype == cp.uint8

    def test_pooling_shapes(self):
        from imgaug2.cuda import pooling as cuda_pooling

        x = cp.asarray(np.random.randint(0, 256, size=(17, 19, 3), dtype=np.uint8))
        y = cuda_pooling.avg_pool(x, 2)
        # Padded to 18x20 -> pooled to 9x10
        assert y.shape == (9, 10, 3)
        assert y.dtype == cp.uint8

    def test_blur_shape_and_dtype(self):
        from imgaug2.cuda import blur as cuda_blur

        x = cp.asarray(np.random.randint(0, 256, size=(16, 20, 3), dtype=np.uint8))
        y = cuda_blur.gaussian_blur(x, sigma=1.0)
        assert y.shape == x.shape
        assert y.dtype == cp.uint8
