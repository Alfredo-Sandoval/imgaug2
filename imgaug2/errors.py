"""Custom exceptions for imgaug2."""

from __future__ import annotations


class ImgaugError(Exception):
    """Base class for imgaug2 exceptions."""


class DependencyMissingError(ImportError, ImgaugError):
    """Raised when an optional dependency is missing."""


class BackendUnavailableError(RuntimeError, ImgaugError):
    """Raised when a backend is installed but not usable on this system."""


class BackendCapabilityError(RuntimeError, ImgaugError):
    """Raised when a backend lacks a required capability."""


__all__ = [
    "BackendCapabilityError",
    "BackendUnavailableError",
    "DependencyMissingError",
    "ImgaugError",
]
