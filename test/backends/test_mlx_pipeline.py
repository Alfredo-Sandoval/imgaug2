import unittest

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False
    mx = None


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxPipeline(unittest.TestCase):
    def test_to_device_and_to_host_roundtrip_uint8(self):
        import imgaug2.mlx as mlx

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        x = mlx.to_device(image)  # default float32
        assert isinstance(x, mx.array)
        assert x.dtype == mx.float32

        out = mlx.to_host(x, dtype=np.uint8)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8
        assert out.shape == image.shape

    def test_chain_keeps_device_between_ops(self):
        import imgaug2.mlx as mlx

        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out = mlx.chain(
            image,
            lambda x: mlx.gaussian_blur(x, sigma=1.0),
            lambda x: mlx.add(x, 1.0),
        )

        assert isinstance(out, np.ndarray)
        assert out.dtype == np.uint8
        assert out.shape == image.shape

    def test_chain_returns_mlx_for_mlx_inputs(self):
        import imgaug2.mlx as mlx

        image = mx.ones((8, 8, 3), dtype=mx.float32)
        out = mlx.chain(image, lambda x: mlx.multiply(x, 2.0))

        assert isinstance(out, mx.array)
        assert out.shape == image.shape
