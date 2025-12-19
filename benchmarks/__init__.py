"""Benchmarking utilities for imgaug2.

This package is intentionally kept separate from the main library code. It is
meant to be executed from the repository root, e.g.:

    python -m benchmarks.runner --platform cpu

Or run the full suite (CPU + legacy + ops + optional third-party baseline):

    python -m benchmarks.run_all
"""

from __future__ import annotations

__all__ = []
