import unittest

try:
    import torch
except Exception:
    torch = None


try:
    import cupy as cp

    from imgaug2.cuda import _core as cuda_core

    CUPY_AVAILABLE = cuda_core.is_available()
except Exception:
    cp = None
    CUPY_AVAILABLE = False


TORCH_CUDA_AVAILABLE = bool(torch is not None and torch.cuda.is_available())


@unittest.skipIf(not (TORCH_CUDA_AVAILABLE and CUPY_AVAILABLE), "torch+cuda and cupy required")
class TestTorchDlpackBridge(unittest.TestCase):
    def test_torch_to_cupy_is_zero_copy_view(self):
        from imgaug2.torch.dlpack import torch_tensor_to_cupy_array

        t = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
        x = torch_tensor_to_cupy_array(t)

        # Modify via CuPy and observe in Torch.
        x += 5.0
        torch.cuda.synchronize()

        expected = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4) + 5.0
        assert torch.allclose(t, expected)

    def test_cupy_to_torch_is_zero_copy_view(self):
        from imgaug2.torch.dlpack import cupy_array_to_torch_tensor

        x = cp.arange(16, dtype=cp.float32).reshape(4, 4)
        t = cupy_array_to_torch_tensor(x)

        # Modify via Torch and observe in CuPy.
        t += 3.0
        torch.cuda.synchronize()

        expected = cp.arange(16, dtype=cp.float32).reshape(4, 4) + 3.0
        cp.testing.assert_allclose(x, expected)
