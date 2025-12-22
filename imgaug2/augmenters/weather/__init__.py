"""Augmenters that create weather effects.

This module provides augmenters that simulate various weather conditions
like snow, clouds, fog, and rain to create realistic environmental effects.

Key Augmenters:
    - `FastSnowyLandscape`: Convert landscapes to snowy scenes.
    - `Clouds`, `CloudLayer`: Add cloud overlays to images.
    - `Fog`: Add fog effects with configurable density.
    - `Snowflakes`, `SnowflakesLayer`: Add falling snowflake effects.
    - `Rain`, `RainLayer`: Add rain streak effects.
"""

from __future__ import annotations

from .clouds import CloudLayer, Clouds, Fog
from .rain import Rain, RainLayer
from .snowflakes import Snowflakes, SnowflakesLayer
from .snowy import FastSnowyLandscape

__all__ = [
    "FastSnowyLandscape",
    "CloudLayer",
    "Clouds",
    "Fog",
    "SnowflakesLayer",
    "Snowflakes",
    "RainLayer",
    "Rain",
]
