from __future__ import annotations

from imgaug2.mlx import router
from imgaug2.mlx._core import is_available


def test_get_routing_info_keys() -> None:
    info = router.get_routing_info("gaussian_blur")
    assert {
        "op",
        "category",
        "min_total_pixels",
        "min_hw",
        "min_batch",
        "prefer_mlx_large_batch",
        "mlx_available",
    }.issubset(info.keys())


def test_get_backend_returns_label() -> None:
    backend = router.get_backend("gaussian_blur", batch=1, height=64, width=64)
    assert backend in {"mlx", "cpu"}


def test_should_use_mlx_returns_bool() -> None:
    decision = router.should_use_mlx("affine_transform", batch=1, height=64, width=64)
    assert isinstance(decision, bool)


def test_estimate_speedup_returns_float() -> None:
    speedup = router.estimate_speedup("gaussian_blur", batch=1, height=64, width=64)
    assert isinstance(speedup, float)
    if not is_available():
        assert speedup == 0.0


def test_update_threshold_roundtrip() -> None:
    original = router.get_routing_info("gaussian_blur")
    router.update_threshold("gaussian_blur", min_total_pixels=12345)
    updated = router.get_routing_info("gaussian_blur")
    assert updated["min_total_pixels"] == 12345
    router.update_threshold(
        "gaussian_blur",
        min_total_pixels=original["min_total_pixels"],
        min_hw=original["min_hw"],
        min_batch=original["min_batch"],
        prefer_mlx_large_batch=original["prefer_mlx_large_batch"],
    )
