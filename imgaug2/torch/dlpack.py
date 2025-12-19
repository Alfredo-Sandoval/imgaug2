"""DLPack-based zero-copy tensor conversion between PyTorch and CuPy.

This module implements bidirectional zero-copy tensor conversion utilities
using the DLPack protocol. DLPack is a standard in-memory tensor structure
that enables efficient data sharing between deep learning frameworks without
copying.

The functions in this module allow seamless integration of PyTorch tensors
with imgaug2's CUDA-accelerated augmentation pipeline by converting between
PyTorch CUDA tensors and CuPy arrays that share the same underlying GPU memory.
"""
from __future__ import annotations

from imgaug2.errors import DependencyMissingError


def torch_tensor_to_cupy_array(tensor: object) -> object:
    """Convert PyTorch CUDA tensor to CuPy array via zero-copy DLPack.

    Creates a CuPy array that shares the same GPU memory as the input PyTorch
    tensor using the DLPack protocol. No data is copied during conversion.

    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor on CUDA device. Must be a CUDA tensor; CPU tensors
        are not supported. Non-contiguous tensors will be made contiguous
        automatically.

    Returns
    -------
    cupy.ndarray
        CuPy array sharing the same GPU memory as the input tensor.

    Raises
    ------
    TypeError
        If `tensor` is not a torch.Tensor instance or if contiguous() is
        unavailable for non-contiguous tensors.
    ValueError
        If `tensor` is not on a CUDA device.
    ImportError
        If PyTorch or CuPy is not installed.
    RuntimeError
        If required DLPack functions are unavailable in torch or cupy.

    Notes
    -----
    - Both PyTorch and CuPy must be installed
    - This function is not imported by default; use explicit import:
      ``from imgaug2.torch.dlpack import torch_tensor_to_cupy_array``
    - Modifications to the returned array will affect the original tensor

    Examples
    --------
    >>> import torch
    >>> tensor = torch.randn(3, 224, 224, device='cuda')
    >>> cupy_array = torch_tensor_to_cupy_array(tensor)
    >>> cupy_array.shape
    (3, 224, 224)
    """
    import importlib

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover
        raise DependencyMissingError("PyTorch is required for torch<->cupy DLPack bridge.") from exc
    tensor_type = getattr(torch, "Tensor", None)

    if tensor_type is None or not isinstance(tensor, tensor_type):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)!r}")

    device = getattr(tensor, "device", None)
    device_type = getattr(device, "type", None)
    if device_type != "cuda":
        raise ValueError(
            "torch_tensor_to_cupy_array requires a CUDA tensor. "
            f"Got tensor.device={device!r}."
        )

    # Ensure contiguous storage for DLPack export
    is_contiguous = getattr(tensor, "is_contiguous", None)
    if callable(is_contiguous) and not is_contiguous():
        contiguous = getattr(tensor, "contiguous", None)
        if not callable(contiguous):
            raise TypeError("Expected torch.Tensor.contiguous()")
        tensor = contiguous()

    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:  # pragma: no cover
        raise DependencyMissingError("CuPy is required for torch<->cupy DLPack bridge.") from exc

    utils = getattr(torch, "utils", None)
    dlpack_mod = getattr(utils, "dlpack", None)
    to_dlpack = getattr(dlpack_mod, "to_dlpack", None)
    if not callable(to_dlpack):
        raise RuntimeError("torch.utils.dlpack.to_dlpack is unavailable")

    dlpack = to_dlpack(tensor)

    # Try both fromDlpack (legacy) and from_dlpack (newer CuPy versions)
    from_dlpack = getattr(cp, "fromDlpack", None)
    if callable(from_dlpack):
        return from_dlpack(dlpack)

    from_dlpack = getattr(cp, "from_dlpack", None)
    if not callable(from_dlpack):
        raise RuntimeError("cupy.fromDlpack/from_dlpack is unavailable")
    return from_dlpack(dlpack)


def cupy_array_to_torch_tensor(array: object) -> object:
    """Convert CuPy array to PyTorch CUDA tensor via zero-copy DLPack.

    Creates a PyTorch CUDA tensor that shares the same GPU memory as the input
    CuPy array using the DLPack protocol. No data is copied during conversion.

    Parameters
    ----------
    array : cupy.ndarray
        CuPy array to convert. Must be a GPU array.

    Returns
    -------
    torch.Tensor
        PyTorch CUDA tensor sharing the same GPU memory as the input array.

    Raises
    ------
    TypeError
        If `array` is not a cupy.ndarray instance.
    ImportError
        If PyTorch or CuPy is not installed.
    RuntimeError
        If PyTorch is not installed, or if required DLPack functions are
        unavailable in torch or cupy.

    Notes
    -----
    - Both PyTorch and CuPy must be installed
    - This function is not imported by default; use explicit import:
      ``from imgaug2.torch.dlpack import cupy_array_to_torch_tensor``
    - Modifications to the returned tensor will affect the original array

    Examples
    --------
    >>> import cupy as cp
    >>> array = cp.random.randn(3, 224, 224)
    >>> tensor = cupy_array_to_torch_tensor(array)
    >>> tensor.shape
    torch.Size([3, 224, 224])
    """
    import importlib

    try:
        cp = importlib.import_module("cupy")
    except Exception as exc:  # pragma: no cover
        raise DependencyMissingError("CuPy is required for torch<->cupy DLPack bridge.") from exc

    ndarray_type = getattr(cp, "ndarray", None)
    if ndarray_type is None or not isinstance(array, ndarray_type):
        raise TypeError(f"Expected cupy.ndarray, got {type(array)!r}")

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover
        raise DependencyMissingError("PyTorch is required for torch<->cupy DLPack bridge.") from exc

    # Use torch.utils.dlpack.from_dlpack with __dlpack__ protocol or explicit capsule
    try:
        utils = getattr(torch, "utils", None)
        dlpack_mod = getattr(utils, "dlpack", None)
        from_dlpack = getattr(dlpack_mod, "from_dlpack", None)
        if not callable(from_dlpack):
            raise RuntimeError("torch.utils.dlpack.from_dlpack is unavailable")
        return from_dlpack(array)
    except Exception as exc:
        to_dlpack = getattr(array, "toDlpack", None)
        if callable(to_dlpack):
            utils = getattr(torch, "utils", None)
            dlpack_mod = getattr(utils, "dlpack", None)
            from_dlpack = getattr(dlpack_mod, "from_dlpack", None)
            if not callable(from_dlpack):
                raise RuntimeError("torch.utils.dlpack.from_dlpack is unavailable") from exc
            return from_dlpack(to_dlpack())
        raise


__all__ = ["torch_tensor_to_cupy_array", "cupy_array_to_torch_tensor"]
