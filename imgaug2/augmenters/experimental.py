"""Experimental augmenters for imgaug2.

This module provides modern augmentation techniques that are common in
state-of-the-art augmentation libraries but were historically missing.

Key Augmenters:
    - `ThinPlateSpline`: Non-linear warping using thin plate splines.
    - `ZoomBlur`: Simulate motion towards/away from camera.
    - `GlassBlur`: Frosted-glass effect using local pixel displacement.
    - `FourierDomainAdaptation`: FDA for domain adaptation.
    - `FancyPCA`: AlexNet-style PCA color jitter.
    - `HEStain`: H&E stain augmentation for histopathology.

Note:
    These augmenters focus on image augmentation; coordinate-based
    augmentable support may be added later.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

import imgaug2.imgaug as ia
import imgaug2.random as iarandom
from imgaug2.augmenters._typing import Array, Images, RNGInput

from . import meta


Number: TypeAlias = float | int
Size: TypeAlias = int | tuple[int, ...] | None
FloatArray: TypeAlias = NDArray[np.floating]
IntArray: TypeAlias = NDArray[np.integer]

Image: TypeAlias = Array


# ---------------------------------------------------------------------
# RNG adapter (works with numpy RandomState, numpy Generator, imgaug RNG)
# ---------------------------------------------------------------------


class _RNGAdapter:
    def __init__(self, rng: object) -> None:
        self._rng = rng

    def random(self, size: Size = None) -> float | FloatArray:
        r = self._rng
        if hasattr(r, "random"):
            return r.random(size)
        if hasattr(r, "rand"):
            if size is None:
                return float(r.rand())
            if isinstance(size, tuple):
                return r.rand(*size)
            return r.rand(size)
        raise AttributeError("RNG has neither `.random()` nor `.rand()`")

    def uniform(
        self, low: Number, high: Number | None = None, size: Size = None
    ) -> float | FloatArray:
        r = self._rng
        if hasattr(r, "uniform"):
            return r.uniform(low, high, size)
        if high is None:
            high = low
            low = 0.0
        return low + (high - low) * self.random(size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Size = None) -> float | FloatArray:
        r = self._rng
        if hasattr(r, "normal"):
            return r.normal(loc, scale, size)
        # Box-Muller
        u1 = self.random(size)
        u2 = self.random(size)
        z0 = np.sqrt(-2.0 * np.log(u1 + 1e-12)) * np.cos(2.0 * np.pi * u2)
        return loc + scale * z0

    def integers(
        self, low: int, high: int | None = None, size: Size = None, endpoint: bool = False
    ) -> int | IntArray:
        r = self._rng
        if hasattr(r, "integers"):
            return r.integers(low, high=high, size=size, endpoint=endpoint)
        if hasattr(r, "randint"):
            if endpoint and high is not None:
                high = high + 1
            return r.randint(low, high=high, size=size)
        raise AttributeError("RNG has neither `.integers()` nor `.randint()`")

    def permutation(self, x: Sequence[object] | Array) -> Array:
        r = self._rng
        if hasattr(r, "permutation"):
            return r.permutation(x)
        x = np.asarray(x)
        idx = np.arange(len(x))
        # Fisher-Yates
        for i in range(len(idx) - 1, 0, -1):
            j = int(self.integers(0, i + 1))
            idx[i], idx[j] = idx[j], idx[i]
        return x[idx]


# ---------------------------------------------------------------------
# CV2 helpers
# ---------------------------------------------------------------------


def _cv2_border_mode(mode: str) -> int:
    mode = mode.lower()
    if mode in ("reflect", "reflect101", "reflect_101", "mirror"):
        return cv2.BORDER_REFLECT_101
    if mode in ("constant", "const"):
        return cv2.BORDER_CONSTANT
    if mode in ("edge", "nearest"):
        return cv2.BORDER_REPLICATE
    if mode in ("wrap",):
        return cv2.BORDER_WRAP
    return cv2.BORDER_REFLECT_101


def _cv2_interpolation(order: int) -> int:
    if order == 0:
        return cv2.INTER_NEAREST
    if order == 1:
        return cv2.INTER_LINEAR
    if order == 2:
        return cv2.INTER_AREA
    if order == 3:
        return cv2.INTER_CUBIC
    if order == 4:
        return cv2.INTER_LANCZOS4
    return cv2.INTER_LINEAR


def _remap_image(
    image: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    order: int = 1,
    mode: str = "reflect",
    cval: float = 0.0,
) -> np.ndarray:
    """Remap an image using a dense coordinate map (output->input)."""
    map_x = map_x.astype(np.float32, copy=False)
    map_y = map_y.astype(np.float32, copy=False)

    out = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=_cv2_interpolation(order),
        borderMode=_cv2_border_mode(mode),
        borderValue=float(cval),
    )
    return out


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    k = int(2 * round(3 * sigma) + 1)
    k = max(3, k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(
        image, (k, k), sigmaX=float(sigma), sigmaY=float(sigma), borderType=cv2.BORDER_REFLECT_101
    )


def _center_crop(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    y0 = max(0, (h - target_h) // 2)
    x0 = max(0, (w - target_w) // 2)
    return image[y0 : y0 + target_h, x0 : x0 + target_w]


def _resize(image: np.ndarray, new_h: int, new_w: int, order: int = 1) -> np.ndarray:
    return cv2.resize(image, (int(new_w), int(new_h)), interpolation=_cv2_interpolation(order))


# ---------------------------------------------------------------------
# TPS (Thin Plate Spline)
# ---------------------------------------------------------------------


def _tps_U(r2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return r2 * np.log(r2 + eps)


def _tps_fit(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Fit TPS params mapping src->dst.

    Returns params with shape (2, N+3): [w (N), a1, ax, ay] for x and y.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    N = src.shape[0]
    diff = src[:, None, :] - src[None, :, :]
    r2 = np.sum(diff**2, axis=2)
    K = _tps_U(r2)
    P = np.concatenate([np.ones((N, 1)), src], axis=1)
    L = np.zeros((N + 3, N + 3), dtype=np.float64)
    L[:N, :N] = K
    L[:N, N:] = P
    L[N:, :N] = P.T
    # tiny regularization for numerical stability
    L[:N, :N] += np.eye(N, dtype=np.float64) * 1e-6

    Yx = np.concatenate([dst[:, 0], np.zeros(3, dtype=np.float64)])
    Yy = np.concatenate([dst[:, 1], np.zeros(3, dtype=np.float64)])
    params_x = np.linalg.solve(L, Yx)
    params_y = np.linalg.solve(L, Yy)
    return np.stack([params_x, params_y], axis=0)


def _tps_eval(
    params: np.ndarray, ctrl: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ctrl = np.asarray(ctrl, dtype=np.float64)
    N = ctrl.shape[0]
    w_x = params[0, :N]
    a_x = params[0, N:]
    w_y = params[1, :N]
    a_y = params[1, N:]

    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)
    pts = np.stack([x_flat, y_flat], axis=1)  # (M,2)
    diff = pts[:, None, :] - ctrl[None, :, :]
    r2 = np.sum(diff**2, axis=2)
    U = _tps_U(r2)

    P = np.stack([np.ones_like(x_flat), x_flat, y_flat], axis=1)
    x_m = U.dot(w_x) + P.dot(a_x)
    y_m = U.dot(w_y) + P.dot(a_y)
    return x_m.reshape(x.shape), y_m.reshape(y.shape)


def _tps_generate_inv_maps(
    height: int,
    width: int,
    rng: _RNGAdapter,
    num_control_points: int,
    scale: float,
    keep_corners: bool,
    map_size: int | tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    # Control points on a grid (pixel coords)
    xs = np.linspace(0, width - 1, num_control_points)
    ys = np.linspace(0, height - 1, num_control_points)
    gx, gy = np.meshgrid(xs, ys)
    ctrl = np.stack([gx.ravel(), gy.ravel()], axis=1)
    N = ctrl.shape[0]

    max_disp = float(scale) * float(min(height, width))
    disp = rng.uniform(-max_disp, max_disp, size=(N, 2))

    if keep_corners:
        idx_tl = 0
        idx_tr = num_control_points - 1
        idx_bl = (num_control_points - 1) * num_control_points
        idx_br = N - 1
        disp[[idx_tl, idx_tr, idx_bl, idx_br], :] = 0.0

    ctrl_disp = ctrl + disp
    ctrl_disp[:, 0] = np.clip(ctrl_disp[:, 0], 0, width - 1)
    ctrl_disp[:, 1] = np.clip(ctrl_disp[:, 1], 0, height - 1)

    # We need output->input mapping for cv2.remap:
    # src=ctrl_disp (output space), dst=ctrl (input space)
    params_inv = _tps_fit(ctrl_disp, ctrl)

    if isinstance(map_size, int):
        mh, mw = map_size, map_size
    else:
        mh, mw = map_size

    x_c = np.linspace(0, width - 1, mw)
    y_c = np.linspace(0, height - 1, mh)
    Xc, Yc = np.meshgrid(x_c, y_c)
    map_x_c, map_y_c = _tps_eval(params_inv, ctrl_disp, Xc, Yc)

    map_x_full = cv2.resize(
        map_x_c.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    map_y_full = cv2.resize(
        map_y_c.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
    )
    return map_x_full, map_y_full


class ThinPlateSpline(meta.Augmenter):
    """
    Thin Plate Spline (TPS) warping.

    Parameters
    ----------
    scale : float or tuple(float, float)
        Displacement magnitude as a fraction of min(H,W). Typical range: 0.01-0.10.
    num_control_points : int
        Control points per axis. Total control points = num_control_points^2.
    keep_corners : bool
        If True, keep the four corner control points fixed (stabilizes the warp).
    map_size : int or (int,int)
        TPS mapping is evaluated on a coarse grid of size map_size and upsampled.
        Larger values are more accurate but slower.
    order : int
        Interpolation order: 0=nearest, 1=linear, 3=cubic.
    mode : str
        Border mode: reflect/constant/edge/wrap.
    cval : number
        Constant border value if mode="constant".
    """

    def __init__(
        self,
        scale: float | tuple[float, float] = (0.01, 0.08),
        num_control_points: int = 5,
        keep_corners: bool = True,
        map_size: int | tuple[int, int] = 32,
        order: int = 1,
        mode: str = "reflect",
        cval: float = 0.0,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.scale = scale
        self.num_control_points = int(num_control_points)
        self.keep_corners = bool(keep_corners)
        self.map_size = map_size
        self.order = int(order)
        self.mode = str(mode)
        self.cval = float(cval)

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            h, w = image.shape[0], image.shape[1]
            if isinstance(self.scale, (tuple, list)):
                scale = float(rng.uniform(self.scale[0], self.scale[1]))
            else:
                scale = float(self.scale)

            map_x, map_y = _tps_generate_inv_maps(
                h,
                w,
                rng,
                num_control_points=self.num_control_points,
                scale=scale,
                keep_corners=self.keep_corners,
                map_size=self.map_size,
            )
            warped = _remap_image(
                image, map_x, map_y, order=self.order, mode=self.mode, cval=self.cval
            )
            result.append(warped)
        return result

    def get_parameters(self) -> list[object]:
        return [
            self.scale,
            self.num_control_points,
            self.keep_corners,
            self.map_size,
            self.order,
            self.mode,
            self.cval,
        ]


# Note: OpticalDistortion is implemented in imgaug2.augmenters.geometric with full augmentable support.
# Access it via imgaug2.augmenters.geometric.OpticalDistortion or imgaug2.augmenters.OpticalDistortion.


# ---------------------------------------------------------------------
# Zoom blur
# ---------------------------------------------------------------------


class ZoomBlur(meta.Augmenter):
    """
    Zoom blur (simulate motion towards/away from camera).

    Parameters
    ----------
    max_factor : float or tuple(float,float)
        Upper bound for zoom factor (>1). 1.2 means up to 20% zoom-in.
    steps : int
        Number of zoom steps averaged.
    order : int
        Resize interpolation order (0/1/3).
    """

    def __init__(
        self,
        max_factor: float | tuple[float, float] = (1.05, 1.30),
        steps: int = 8,
        order: int = 1,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.max_factor = max_factor
        self.steps = int(steps)
        self.order = int(order)

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            h, w = image.shape[0], image.shape[1]
            if isinstance(self.max_factor, (tuple, list)):
                max_factor = float(rng.uniform(self.max_factor[0], self.max_factor[1]))
            else:
                max_factor = float(self.max_factor)

            factors = np.linspace(1.0, max_factor, self.steps, dtype=np.float32)
            acc = np.zeros_like(image, dtype=np.float32)
            for f in factors:
                new_h = max(1, int(round(h * float(f))))
                new_w = max(1, int(round(w * float(f))))
                resized = _resize(image, new_h, new_w, order=self.order)
                cropped = _center_crop(resized, h, w)
                acc += cropped.astype(np.float32)
            out = acc / float(len(factors))
            if np.issubdtype(image.dtype, np.integer):
                out = np.clip(out, 0, np.iinfo(image.dtype).max).astype(image.dtype)
            else:
                out = out.astype(image.dtype)
            result.append(out)
        return result

    def get_parameters(self) -> list[object]:
        return [self.max_factor, self.steps, self.order]


# ---------------------------------------------------------------------
# Glass blur
# ---------------------------------------------------------------------


class GlassBlur(meta.Augmenter):
    """
    Glass blur (frosted-glass effect).

    Approximated as:
      Gaussian blur -> random local pixel displacement -> Gaussian blur

    Parameters
    ----------
    sigma : float or tuple(float,float)
        Sigma for Gaussian blur.
    max_delta : int
        Maximum pixel displacement in x/y.
    iterations : int
        Number of displacement iterations.
    swap_fraction : float
        Fraction of pixels displaced per iteration (0..1]. 1.0 is strongest.
    """

    def __init__(
        self,
        sigma: float | tuple[float, float] = (0.5, 1.5),
        max_delta: int = 2,
        iterations: int = 2,
        swap_fraction: float = 0.25,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.sigma = sigma
        self.max_delta = int(max_delta)
        self.iterations = int(iterations)
        self.swap_fraction = float(swap_fraction)

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            if isinstance(self.sigma, (tuple, list)):
                sigma = float(rng.uniform(self.sigma[0], self.sigma[1]))
            else:
                sigma = float(self.sigma)

            img_blur = _gaussian_blur(image, sigma)
            h, w = img_blur.shape[0], img_blur.shape[1]
            md = int(self.max_delta)
            if md <= 0 or h <= 2 * md or w <= 2 * md:
                result.append(img_blur)
                continue

            n_pixels = int(round((h - 2 * md) * (w - 2 * md) * float(self.swap_fraction)))
            n_pixels = max(0, n_pixels)
            out = img_blur
            for _ in range(self.iterations):
                if n_pixels <= 0:
                    break
                ys = rng.integers(md, h - md, size=n_pixels)
                xs = rng.integers(md, w - md, size=n_pixels)
                dx = rng.integers(-md, md + 1, size=n_pixels)
                dy = rng.integers(-md, md + 1, size=n_pixels)
                ys2 = ys + dy
                xs2 = xs + dx
                displaced = out.copy()
                displaced[ys, xs] = out[ys2, xs2]
                out = _gaussian_blur(displaced, sigma)

            result.append(out)
        return result

    def get_parameters(self) -> list[object]:
        return [self.sigma, self.max_delta, self.iterations, self.swap_fraction]


# ---------------------------------------------------------------------
# FDA (Fourier Domain Adaptation)
# ---------------------------------------------------------------------


def _fda_image(source: np.ndarray, target: np.ndarray, beta: float) -> np.ndarray:
    src = source.astype(np.float32)
    tgt = target.astype(np.float32)
    h, w = src.shape[0], src.shape[1]
    if tgt.shape[0] != h or tgt.shape[1] != w:
        tgt = _resize(tgt, h, w, order=1)

    if src.ndim == 2:
        src_c = src[..., None]
        tgt_c = tgt[..., None] if tgt.ndim == 2 else tgt[..., :1]
    else:
        src_c = src
        if tgt.ndim == 2:
            tgt_c = np.repeat(tgt[..., None], src_c.shape[2], axis=2)
        else:
            tgt_c = tgt[..., : src_c.shape[2]]

    C = src_c.shape[2]
    b = max(0.0, min(0.5, float(beta)))
    L = int(round(min(h, w) * b))
    cy, cx = h // 2, w // 2
    y1, y2 = cy - L, cy + L + 1
    x1, x2 = cx - L, cx + L + 1

    out = np.empty_like(src_c, dtype=np.float32)
    for c in range(C):
        fs = np.fft.fft2(src_c[..., c])
        ft = np.fft.fft2(tgt_c[..., c])
        fs_shift = np.fft.fftshift(fs)
        ft_shift = np.fft.fftshift(ft)

        amp_s = np.abs(fs_shift)
        phase_s = np.angle(fs_shift)
        amp_t = np.abs(ft_shift)

        amp_s[y1:y2, x1:x2] = amp_t[y1:y2, x1:x2]
        fs_new = np.fft.ifftshift(amp_s * np.exp(1j * phase_s))
        img_new = np.fft.ifft2(fs_new)
        out[..., c] = np.real(img_new).astype(np.float32)

    out_img = out[..., 0] if source.ndim == 2 else out
    if np.issubdtype(source.dtype, np.integer):
        out_img = np.clip(out_img, 0, np.iinfo(source.dtype).max).astype(source.dtype)
    else:
        out_img = out_img.astype(source.dtype)
    return out_img


class FourierDomainAdaptation(meta.Augmenter):
    """
    Fourier Domain Adaptation (FDA).

    Parameters
    ----------
    reference_images : sequence of ndarray
        One or more reference images to sample from.
    beta : float or tuple(float,float)
        Fraction controlling swapped low-frequency region size.
    """

    def __init__(
        self,
        reference_images: Sequence[Image] | None,
        beta: float | tuple[float, float] = (0.01, 0.10),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        if reference_images is None or len(reference_images) == 0:
            raise ValueError("reference_images must be a non-empty sequence of images.")
        self.reference_images = list(reference_images)
        self.beta = beta

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            if isinstance(self.beta, (tuple, list)):
                beta = float(rng.uniform(self.beta[0], self.beta[1]))
            else:
                beta = float(self.beta)
            idx = int(rng.integers(0, len(self.reference_images)))
            ref = self.reference_images[idx]
            result.append(_fda_image(image, ref, beta=beta))
        return result

    def get_parameters(self) -> list[object]:
        return [len(self.reference_images), self.beta]


# ---------------------------------------------------------------------
# FancyPCA
# ---------------------------------------------------------------------


class FancyPCA(meta.Augmenter):
    """
    Fancy PCA color augmentation (AlexNet-style PCA jitter).

    Parameters
    ----------
    alpha_std : float or tuple(float,float)
        Stddev for PCA coefficient sampling. Common: 0.1.
    eigvecs, eigvals : Optional precomputed PCA decomposition.
        If provided, PCA is not recomputed per image.
    """

    def __init__(
        self,
        alpha_std: float | tuple[float, float] = (0.0, 0.1),
        eigvecs: np.ndarray | None = None,
        eigvals: np.ndarray | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.alpha_std = alpha_std
        self.eigvecs = eigvecs
        self.eigvals = eigvals

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            if image.ndim != 3 or image.shape[2] < 3:
                result.append(image)
                continue

            if isinstance(self.alpha_std, (tuple, list)):
                alpha_std = float(rng.uniform(self.alpha_std[0], self.alpha_std[1]))
            else:
                alpha_std = float(self.alpha_std)

            img = image.astype(np.float32)
            rgb = img[..., :3]
            flat = rgb.reshape(-1, 3)

            if self.eigvecs is None or self.eigvals is None:
                mean = np.mean(flat, axis=0)
                X = flat - mean
                cov = np.cov(X, rowvar=False)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]
                eigvals = eigvals[order]
                eigvecs = eigvecs[:, order]
            else:
                eigvecs = np.asarray(self.eigvecs, dtype=np.float32)
                eigvals = np.asarray(self.eigvals, dtype=np.float32).reshape(3)

            alphas = rng.normal(0.0, alpha_std, size=3).astype(np.float32)
            delta = (eigvecs * (alphas * eigvals)).sum(axis=1)

            out = img.copy()
            out[..., :3] = rgb + delta[None, None, :]

            if np.issubdtype(image.dtype, np.integer):
                out = np.clip(out, 0, np.iinfo(image.dtype).max).astype(image.dtype)
            else:
                out = out.astype(image.dtype)
            result.append(out)
        return result

    def get_parameters(self) -> list[object]:
        return [self.alpha_std]


# ---------------------------------------------------------------------
# H&E stain jitter
# ---------------------------------------------------------------------


class HEStain(meta.Augmenter):
    """
    H&E stain augmentation.

    Converts RGB -> optical density -> deconvolution -> perturb stain channels -> reconvolution.

    Parameters
    ----------
    alpha : tuple(float,float)
        Multiplicative jitter for stain concentrations.
    beta : tuple(float,float)
        Additive jitter for stain concentrations.
    stain_matrix : Optional ndarray (3,2) or (3,3)
        Custom stain matrix columns.
    """

    def __init__(
        self,
        alpha: tuple[float, float] = (0.9, 1.1),
        beta: tuple[float, float] = (-0.02, 0.02),
        stain_matrix: np.ndarray | None = None,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.alpha = (float(alpha[0]), float(alpha[1]))
        self.beta = (float(beta[0]), float(beta[1]))
        self.stain_matrix = stain_matrix

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            if image.ndim != 3 or image.shape[2] < 3:
                result.append(image)
                continue

            img = image.astype(np.float32)

            if self.stain_matrix is None:
                H = np.array([0.650, 0.704, 0.286], dtype=np.float32)
                E = np.array([0.072, 0.990, 0.105], dtype=np.float32)
                R = np.cross(H, E)
                M = np.stack([H, E, R], axis=1)
            else:
                M = np.asarray(self.stain_matrix, dtype=np.float32)
                if M.shape == (3, 2):
                    R = np.cross(M[:, 0], M[:, 1])
                    M = np.concatenate([M, R[:, None]], axis=1)
                if M.shape != (3, 3):
                    raise ValueError("stain_matrix must be (3,2) or (3,3)")

            M = M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-8)

            OD = -np.log((img[..., :3] + 1.0) / 255.0)
            OD_flat = OD.reshape(-1, 3).T
            C = np.linalg.solve(M, OD_flat)

            a_h = float(rng.uniform(self.alpha[0], self.alpha[1]))
            a_e = float(rng.uniform(self.alpha[0], self.alpha[1]))
            b_h = float(rng.uniform(self.beta[0], self.beta[1]))
            b_e = float(rng.uniform(self.beta[0], self.beta[1]))

            C_aug = C.copy()
            C_aug[0, :] = C_aug[0, :] * a_h + b_h
            C_aug[1, :] = C_aug[1, :] * a_e + b_e

            OD_new = (M @ C_aug).T.reshape(OD.shape)
            rgb_new = 255.0 * np.exp(-OD_new) - 1.0

            out = img.copy()
            out[..., :3] = rgb_new

            if np.issubdtype(image.dtype, np.integer):
                out = np.clip(out, 0, np.iinfo(image.dtype).max).astype(image.dtype)
            else:
                out = out.astype(image.dtype)
            result.append(out)
        return result

    def get_parameters(self) -> list[object]:
        return [self.alpha, self.beta]


# ---------------------------------------------------------------------
# Planckian jitter (white balance via color temperature)
# ---------------------------------------------------------------------


def _temperature_to_rgb(temp_k: float) -> np.ndarray:
    """
    Approximate black-body color temperature -> RGB multipliers (0..1).
    """
    t = float(temp_k) / 100.0

    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)

    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)

    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307

    rgb = np.array([r, g, b], dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    return rgb


class PlanckianJitter(meta.Augmenter):
    """
    Planckian jitter: illumination color change via temperature shift.

    Parameters
    ----------
    temperature : tuple(float,float)
        Sampled Kelvin temperature range (approx 1000-40000K is reasonable).
    reference_temp : float
        Reference temperature used to compute channel scaling (default 6500K).
    strength : float or tuple(float,float)
        Blend factor between identity (0) and full scaling (1).
    """

    def __init__(
        self,
        temperature: tuple[float, float] = (2500.0, 10000.0),
        reference_temp: float = 6500.0,
        strength: float | tuple[float, float] = (0.5, 1.0),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.temperature = (float(temperature[0]), float(temperature[1]))
        self.reference_temp = float(reference_temp)
        self.strength = strength

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        rgb_ref = _temperature_to_rgb(self.reference_temp)
        result: list[Image] = []
        for image in images:
            if image.ndim != 3 or image.shape[2] < 3:
                result.append(image)
                continue
            t = float(rng.uniform(self.temperature[0], self.temperature[1]))
            rgb_t = _temperature_to_rgb(t)
            scale = rgb_t / (rgb_ref + 1e-8)

            if isinstance(self.strength, (tuple, list)):
                s = float(rng.uniform(self.strength[0], self.strength[1]))
            else:
                s = float(self.strength)
            scale = 1.0 + (scale - 1.0) * s

            img = image.astype(np.float32)
            out = img.copy()
            out[..., :3] = img[..., :3] * scale[None, None, :]
            if np.issubdtype(image.dtype, np.integer):
                out = np.clip(out, 0, np.iinfo(image.dtype).max).astype(image.dtype)
            else:
                out = out.astype(image.dtype)
            result.append(out)
        return result

    def get_parameters(self) -> list[object]:
        return [self.temperature, self.reference_temp, self.strength]


# ---------------------------------------------------------------------
# Plasma effects (diamond-square fractal)
# ---------------------------------------------------------------------


def _next_pow2(n: int) -> int:
    return 1 << (int(n) - 1).bit_length()


def _plasma_fractal(height: int, width: int, rng: _RNGAdapter, roughness: float) -> np.ndarray:
    size = max(height, width)
    s = _next_pow2(size - 1) + 1  # 2^k + 1
    grid = np.zeros((s, s), dtype=np.float32)

    grid[0, 0] = float(rng.random())
    grid[0, -1] = float(rng.random())
    grid[-1, 0] = float(rng.random())
    grid[-1, -1] = float(rng.random())

    step = s - 1
    scale = 1.0
    roughness = float(roughness)

    while step > 1:
        half = step // 2

        # diamond step
        for y in range(half, s - 1, step):
            for x in range(half, s - 1, step):
                avg = (
                    grid[y - half, x - half]
                    + grid[y - half, x + half]
                    + grid[y + half, x - half]
                    + grid[y + half, x + half]
                ) / 4.0
                grid[y, x] = avg + (float(rng.random()) - 0.5) * scale

        # square step
        for y in range(0, s, half):
            for x in range((y + half) % step, s, step):
                vals = []
                if y - half >= 0:
                    vals.append(grid[y - half, x])
                if y + half < s:
                    vals.append(grid[y + half, x])
                if x - half >= 0:
                    vals.append(grid[y, x - half])
                if x + half < s:
                    vals.append(grid[y, x + half])
                avg = float(sum(vals)) / float(len(vals))
                grid[y, x] = avg + (float(rng.random()) - 0.5) * scale

        step = half
        scale *= roughness

    gmin, gmax = float(grid.min()), float(grid.max())
    if gmax > gmin:
        grid = (grid - gmin) / (gmax - gmin)

    if grid.shape[0] != height or grid.shape[1] != width:
        grid = cv2.resize(grid, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
    return grid.astype(np.float32)


class PlasmaBrightness(meta.Augmenter):
    """
    Plasma brightness: add structured (fractal) brightness variations.

    Parameters
    ----------
    intensity : tuple(float,float)
        Strength in [0,1] where 1 corresponds to large swings (scaled to 255).
    roughness : tuple(float,float)
        Diamond-square roughness (0<r<1). Lower -> smoother.
    """

    def __init__(
        self,
        intensity: tuple[float, float] = (0.1, 0.5),
        roughness: tuple[float, float] = (0.6, 0.9),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.intensity = (float(intensity[0]), float(intensity[1]))
        self.roughness = (float(roughness[0]), float(roughness[1]))

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            h, w = image.shape[0], image.shape[1]
            r = float(rng.uniform(self.roughness[0], self.roughness[1]))
            mask = _plasma_fractal(h, w, rng, roughness=r)
            amp = float(rng.uniform(self.intensity[0], self.intensity[1]))
            delta = (mask - 0.5) * (amp * 255.0 * 2.0)

            img = image.astype(np.float32)
            out = img.copy()
            if out.ndim == 2:
                out = out + delta
            else:
                out[..., :3] = out[..., :3] + delta[..., None]

            if np.issubdtype(image.dtype, np.integer):
                out = np.clip(out, 0, np.iinfo(image.dtype).max).astype(image.dtype)
            else:
                out = out.astype(image.dtype)
            result.append(out)
        return result

    def get_parameters(self) -> list[object]:
        return [self.intensity, self.roughness]


class PlasmaShadow(meta.Augmenter):
    """
    Plasma shadow: multiply image by a structured (fractal) darkening mask.

    Parameters
    ----------
    intensity : tuple(float,float)
        Strength in [0,1]. Higher -> darker shadows.
    roughness : tuple(float,float)
        Diamond-square roughness (0<r<1). Lower -> smoother.
    """

    def __init__(
        self,
        intensity: tuple[float, float] = (0.2, 0.8),
        roughness: tuple[float, float] = (0.6, 0.9),
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )
        self.intensity = (float(intensity[0]), float(intensity[1]))
        self.roughness = (float(roughness[0]), float(roughness[1]))

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        rng = _RNGAdapter(random_state)
        result: list[Image] = []
        for image in images:
            h, w = image.shape[0], image.shape[1]
            r = float(rng.uniform(self.roughness[0], self.roughness[1]))
            mask = _plasma_fractal(h, w, rng, roughness=r)
            strength = float(rng.uniform(self.intensity[0], self.intensity[1]))
            mult = 1.0 - strength * mask

            img = image.astype(np.float32)
            out = img.copy()
            if out.ndim == 2:
                out = out * mult
            else:
                out[..., :3] = out[..., :3] * mult[..., None]

            if np.issubdtype(image.dtype, np.integer):
                out = np.clip(out, 0, np.iinfo(image.dtype).max).astype(image.dtype)
            else:
                out = out.astype(image.dtype)
            result.append(out)
        return result

    def get_parameters(self) -> list[object]:
        return [self.intensity, self.roughness]
