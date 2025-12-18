"""Dummy MLX core to satisfy imports."""
from __future__ import annotations
from typing import Any

def is_mlx_array(obj: object) -> bool:
    """Always returns False as MLX is not available."""
    return False

def to_numpy(arr: Any) -> Any:
    """Dummy to_numpy."""
    return arr

class Dummy:
    """Dummy object to satisfy attribute access."""
    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None

mx = Dummy()
geometry = Dummy()
