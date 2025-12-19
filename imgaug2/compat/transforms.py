"""Transform wrappers for dict-based compatibility API.

This module provides Albumentations-compatible transform classes that wrap
imgaug2 augmenters. Each transform accepts a probability parameter and converts
to an imgaug2 augmenter when called. Compose handles probability application.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import imgaug2.augmenters as iaa


def _ensure_tuple2(val: float | tuple[float, float]) -> tuple[float, float]:
    """Convert value to symmetric range tuple.

    Parameters
    ----------
    val : float or tuple of float
        Single value or (min, max) tuple.

    Returns
    -------
    tuple of float
        Range as (min, max). Single values map to (-val, val).

    Raises
    ------
    ValueError
        If tuple length is not 2.
    """
    if isinstance(val, tuple):
        if len(val) != 2:
            raise ValueError(f"Expected tuple of length 2, got {val!r}")
        return float(val[0]), float(val[1])
    return -float(val), float(val)


def _maybe_tuple(val: float | tuple[float, float]) -> tuple[float, float]:
    """Convert value to tuple, preserving single values.

    Parameters
    ----------
    val : float or tuple of float
        Single value or (min, max) tuple.

    Returns
    -------
    tuple of float
        Range as (min, max). Single values map to (val, val).

    Raises
    ------
    ValueError
        If tuple length is not 2.
    """
    if isinstance(val, tuple):
        if len(val) != 2:
            raise ValueError(f"Expected tuple of length 2, got {val!r}")
        return float(val[0]), float(val[1])
    return float(val), float(val)


def _to_affine_mode(border_mode: str | int) -> str:
    """Convert border mode to imgaug2 Affine mode string.

    Parameters
    ----------
    border_mode : str or int
        Border mode as string or OpenCV constant.

    Returns
    -------
    str
        Mode string for imgaug2.Affine ('constant', 'edge', etc.).

    Raises
    ------
    ValueError
        If border_mode is unsupported.
    """
    if isinstance(border_mode, str):
        return border_mode

    mapping = {
        cv2.BORDER_CONSTANT: "constant",
        cv2.BORDER_REPLICATE: "edge",
        cv2.BORDER_REFLECT: "symmetric",
        cv2.BORDER_REFLECT_101: "reflect",
        cv2.BORDER_WRAP: "wrap",
    }
    if border_mode not in mapping:
        raise ValueError(f"Unsupported border_mode: {border_mode!r}")
    return mapping[border_mode]


@dataclass(frozen=True, slots=True)
class BasicTransform:
    """Base class for all compatibility transforms.

    Parameters
    ----------
    p : float, default=0.5
        Probability of applying the transform.

    Notes
    -----
    Subclasses must implement to_iaa() method to convert to imgaug2 augmenter.
    Compose uses p for probabilistic application via iaa.Sometimes.
    """

    p: float = 0.5

    def to_iaa(self) -> iaa.Augmenter:
        """Convert transform to imgaug2 augmenter.

        Returns
        -------
        iaa.Augmenter
            Equivalent imgaug2 augmenter (always-apply, probability handled by Compose).

        Raises
        ------
        NotImplementedError
            If subclass doesn't implement this method.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class HorizontalFlip(BasicTransform):
    """Flip image horizontally (left to right).

    Parameters
    ----------
    p : float, default=0.5
        Probability of applying the transform.
    """

    def to_iaa(self) -> iaa.Augmenter:
        """Convert to imgaug2 horizontal flip augmenter.

        Returns
        -------
        iaa.Augmenter
            Fliplr augmenter with probability 1.0.
        """
        return iaa.Fliplr(1.0)


@dataclass(frozen=True, slots=True)
class VerticalFlip(BasicTransform):
    """Flip image vertically (top to bottom).

    Parameters
    ----------
    p : float, default=0.5
        Probability of applying the transform.
    """

    def to_iaa(self) -> iaa.Augmenter:
        """Convert to imgaug2 vertical flip augmenter.

        Returns
        -------
        iaa.Augmenter
            Flipud augmenter with probability 1.0.
        """
        return iaa.Flipud(1.0)


@dataclass(frozen=True, slots=True)
class Rotate(BasicTransform):
    """Rotate image by random angle.

    Parameters
    ----------
    limit : float or tuple of float, default=90.0
        Rotation range in degrees. Single value creates symmetric range (-limit, limit).
        Tuple specifies (min, max) range.
    border_mode : str or int, default='reflect'
        Border handling mode (see cv2.BORDER_* or imgaug2 mode strings).
    value : float or int, default=0
        Fill value for constant border mode.
    p : float, default=0.5
        Probability of applying the transform.
    """

    limit: float | tuple[float, float] = 90.0
    border_mode: str | int = "reflect"
    value: float | int = 0

    def to_iaa(self) -> iaa.Augmenter:
        """Convert to imgaug2 rotation augmenter.

        Returns
        -------
        iaa.Augmenter
            Affine augmenter with rotation.
        """
        low, high = _ensure_tuple2(self.limit)
        return iaa.Affine(
            rotate=(low, high), mode=_to_affine_mode(self.border_mode), cval=self.value
        )


@dataclass(frozen=True, slots=True)
class ShiftScaleRotate(BasicTransform):
    """Apply random shift, scale, and rotation.

    Parameters
    ----------
    shift_limit : float or tuple of float, default=0.0625
        Translation range as fraction of image size. Single value creates
        symmetric range (-limit, limit).
    scale_limit : float or tuple of float, default=0.1
        Scaling factor range. Single value creates range (1-limit, 1+limit).
        Tuple is interpreted as (1+a, 1+b).
    rotate_limit : float or tuple of float, default=45.0
        Rotation range in degrees. Single value creates symmetric range.
    border_mode : str or int, default='reflect'
        Border handling mode.
    value : float or int, default=0
        Fill value for constant border mode.
    p : float, default=0.5
        Probability of applying the transform.
    """

    shift_limit: float | tuple[float, float] = 0.0625
    scale_limit: float | tuple[float, float] = 0.1
    rotate_limit: float | tuple[float, float] = 45.0
    border_mode: str | int = "reflect"
    value: float | int = 0

    def to_iaa(self) -> iaa.Augmenter:
        """Convert to imgaug2 combined affine transform.

        Returns
        -------
        iaa.Augmenter
            Affine augmenter with shift, scale, and rotation.
        """
        shift_low, shift_high = _ensure_tuple2(self.shift_limit)

        if isinstance(self.scale_limit, tuple):
            a, b = _maybe_tuple(self.scale_limit)
            scale = (1.0 + a, 1.0 + b)
        else:
            scale = (1.0 - float(self.scale_limit), 1.0 + float(self.scale_limit))

        rot_low, rot_high = _ensure_tuple2(self.rotate_limit)

        return iaa.Affine(
            scale=scale,
            translate_percent={"x": (shift_low, shift_high), "y": (shift_low, shift_high)},
            rotate=(rot_low, rot_high),
            mode=_to_affine_mode(self.border_mode),
            cval=self.value,
        )


@dataclass(frozen=True, slots=True)
class RandomBrightnessContrast(BasicTransform):
    """Apply random brightness and contrast adjustments.

    Parameters
    ----------
    brightness_limit : float or tuple of float, default=0.2
        Brightness adjustment range. Single value creates symmetric range
        [-limit*255, limit*255]. Tuple is [low*255, high*255].
    contrast_limit : float or tuple of float, default=0.2
        Contrast multiplier range. Single value creates range [1-limit, 1+limit].
        Tuple is [1+low, 1+high].
    p : float, default=0.5
        Probability of applying the transform.

    Notes
    -----
    Brightness is applied as additive offset, contrast as multiplicative factor.
    Operations are applied in order: multiply then add.
    """

    brightness_limit: float | tuple[float, float] = 0.2
    contrast_limit: float | tuple[float, float] = 0.2

    def to_iaa(self) -> iaa.Augmenter:
        """Convert to imgaug2 brightness/contrast augmenter.

        Returns
        -------
        iaa.Augmenter
            Sequential augmenter with Multiply then Add.
        """
        if isinstance(self.brightness_limit, tuple):
            bl, bh = _maybe_tuple(self.brightness_limit)
            add = (bl * 255.0, bh * 255.0)
        else:
            b = float(self.brightness_limit)
            add = (-b * 255.0, b * 255.0)

        if isinstance(self.contrast_limit, tuple):
            cl, ch = _maybe_tuple(self.contrast_limit)
            mul = (1.0 + cl, 1.0 + ch)
        else:
            c = float(self.contrast_limit)
            mul = (1.0 - c, 1.0 + c)

        return iaa.Sequential([iaa.Multiply(mul), iaa.Add(add)])
