"""Metal kernels for MLX fast paths.

Internal implementation details for blur/warp ops. Inputs are NHWC MLX arrays,
and outputs stay on device. Not part of the public API surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from imgaug2.errors import BackendCapabilityError
from ._core import mx, require

if TYPE_CHECKING:
    from mlx.core import array as MxArray
else:
    MxArray = object


class _MetalKernel(Protocol):
    def __call__(
        self,
        *,
        inputs: Sequence[MxArray],
        template: Sequence[tuple[str, object]],
        grid: tuple[int, int, int],
        threadgroup: tuple[int, int, int],
        output_shapes: Sequence[tuple[int, ...]],
        output_dtypes: Sequence[object],
    ) -> list[MxArray]: ...


def _require_metal() -> None:
    require()
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        raise BackendCapabilityError(
            "MLX is available, but mx.fast.metal_kernel is not present in this MLX build. "
            "Upgrade `mlx` to a version that includes Metal kernels."
        )


@lru_cache(maxsize=128)
def _out_hw_mx(out_h: int, out_w: int) -> MxArray:
    _require_metal()
    return mx.array(np.array([int(out_h), int(out_w)], dtype=np.int32))


_METAL_HEADER_REFLECT101 = r"""
inline float reflect101_f(float v, int size) {
    if (size <= 1) return 0.0f;
    float period = 2.0f * (float)(size - 1);
    v = metal::fabs(v);
    v = metal::fmod(v, period);
    return (v > (float)(size - 1)) ? (period - v) : v;
}

// Fast reflect101 for small out-of-range integer indices.
inline int reflect101_i_fast(int v, int size) {
    if (size <= 1) return 0;

    if (v < 0) v = -v;
    if (v >= size) v = 2 * size - v - 2;

    if (v < 0) v = -v;
    if (v >= size) v = 2 * size - v - 2;

    return clamp(v, 0, size - 1);
}

// Correct reflect101 for any integer index (uses modulo).
inline int reflect101_i_mod(int v, int size) {
    if (size <= 1) return 0;
    int period = 2 * (size - 1);
    v = abs(v);
    v = v % period;
    if (v >= size) v = period - v;
    return clamp(v, 0, size - 1);
}
"""


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _tile_for_ksize_tg32(ksize: int) -> int:
    r = ksize // 2
    return 32 - 2 * r


# Threshold for switching between channel-parallel and pixel-parallel kernels.
# - Channel-parallel: more GPU threads (N*H*W*C), simpler work per thread
# - Pixel-parallel: fewer threads (N*H*W), loops over channels internally
#
# For small images, channel-parallel has better parallelism. For large images,
# pixel-parallel reduces thread divergence and memory traffic.
#
# Tuned based on benchmarks: 256x256 = 65,536 keeps channel-parallel for
# single medium-sized images where it's often faster.
_WARP_CHANNEL_PARALLEL_THRESHOLD = 256 * 256


@lru_cache(maxsize=4)
def _warp_affine_kernel_pixel() -> _MetalKernel:
    _require_metal()

    source = r"""
        uint tid = thread_position_in_grid.x;

        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];

        int outH = out_hw[0];
        int outW = out_hw[1];

        int outHW = outH * outW;
        int b = (int)(tid / (uint)outHW);
        int t = (int)(tid - (uint)(b * outHW));
        int oy = t / outW;
        int ox = t - oy * outW;

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        int out_w_stride = C;
        int out_h_stride = outW * out_w_stride;
        int out_b_stride = outH * out_h_stride;

        int out_base = b * out_b_stride + oy * out_h_stride + ox * out_w_stride;

        float m00 = M[0];
        float m01 = M[1];
        float m02 = M[2];
        float m10 = M[3];
        float m11 = M[4];
        float m12 = M[5];

        float sx = m00 * (float)ox + m01 * (float)oy + m02;
        float sy = m10 * (float)ox + m11 * (float)oy + m12;

        int pad_mode = pad_mode_i[0]; // 0=zeros, 1=border, 2=reflect101
        int interp_mode = interp_mode_i[0]; // 0=nearest, 1=bilinear
        float cval = cval_f[0];

        if (pad_mode == 2) {
            sx = reflect101_f(sx, W);
            sy = reflect101_f(sy, H);
        } else if (pad_mode == 1) {
            sx = clamp(sx, 0.0f, (float)(W - 1));
            sy = clamp(sy, 0.0f, (float)(H - 1));
        }

        int base_b = b * b_stride;

        if (interp_mode == 0) {
            int xn = (int)metal::round(sx);
            int yn = (int)metal::round(sy);
            if (pad_mode == 0) {
                if (xn < 0 || xn >= W || yn < 0 || yn >= H) {
                    for (int c = 0; c < C; ++c) {
                        out[out_base + c] = (T)cval;
                    }
                } else {
                    int off = base_b + yn * h_stride + xn * w_stride;
                    for (int c = 0; c < C; ++c) {
                        out[out_base + c] = x[off + c];
                    }
                }
            } else {
                xn = clamp(xn, 0, W - 1);
                yn = clamp(yn, 0, H - 1);
                int off = base_b + yn * h_stride + xn * w_stride;
                for (int c = 0; c < C; ++c) {
                    out[out_base + c] = x[off + c];
                }
            }
            return;
        }

        int x0 = (int)metal::floor(sx);
        int y0 = (int)metal::floor(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float wx = sx - (float)x0;
        float wy = sy - (float)y0;

        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = wx * (1.0f - wy);
        float w10 = (1.0f - wx) * wy;
        float w11 = wx * wy;

        int x0c = clamp(x0, 0, W - 1);
        int y0c = clamp(y0, 0, H - 1);
        int x1c = clamp(x1, 0, W - 1);
        int y1c = clamp(y1, 0, H - 1);

        int off00 = base_b + y0c * h_stride + x0c * w_stride;
        int off01 = base_b + y0c * h_stride + x1c * w_stride;
        int off10 = base_b + y1c * h_stride + x0c * w_stride;
        int off11 = base_b + y1c * h_stride + x1c * w_stride;

        bool in00 = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H);
        bool in01 = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H);
        bool in10 = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H);
        bool in11 = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H);

        for (int c = 0; c < C; ++c) {
            float v00 = (pad_mode == 0 && !in00) ? cval : (float)x[off00 + c];
            float v01 = (pad_mode == 0 && !in01) ? cval : (float)x[off01 + c];
            float v10 = (pad_mode == 0 && !in10) ? cval : (float)x[off10 + c];
            float v11 = (pad_mode == 0 && !in11) ? cval : (float)x[off11 + c];
            float outv = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
            out[out_base + c] = (T)outv;
        }
    """

    return cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name="imgaug2_warp_affine_bilinear_reflect101_pixel",
            input_names=["x", "M", "out_hw", "pad_mode_i", "interp_mode_i", "cval_f"],
            output_names=["out"],
            source=source,
            header=_METAL_HEADER_REFLECT101,
            ensure_row_contiguous=True,
        ),
    )


@lru_cache(maxsize=4)
def _warp_affine_kernel_channel() -> _MetalKernel:
    _require_metal()

    source = r"""
        uint elem = thread_position_in_grid.x;

        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];

        int outH = out_hw[0];
        int outW = out_hw[1];

        int outHW = outH * outW;

        int pix = (int)(elem / (uint)C);
        int c = (int)(elem - (uint)(pix * C));
        int b = pix / outHW;
        int t = pix - b * outHW;
        int oy = t / outW;
        int ox = t - oy * outW;

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        int base_idx = b * b_stride + c;
        int out_idx = (b * outHW + t) * C + c;

        float m00 = M[0];
        float m01 = M[1];
        float m02 = M[2];
        float m10 = M[3];
        float m11 = M[4];
        float m12 = M[5];

        float sx = m00 * (float)ox + m01 * (float)oy + m02;
        float sy = m10 * (float)ox + m11 * (float)oy + m12;

        int pad_mode = pad_mode_i[0]; // 0=zeros, 1=border, 2=reflect101
        int interp_mode = interp_mode_i[0]; // 0=nearest, 1=bilinear
        float cval = cval_f[0];

        if (pad_mode == 2) {
            sx = reflect101_f(sx, W);
            sy = reflect101_f(sy, H);
        } else if (pad_mode == 1) {
            sx = clamp(sx, 0.0f, (float)(W - 1));
            sy = clamp(sy, 0.0f, (float)(H - 1));
        }

        if (interp_mode == 0) {
            int xn = (int)metal::round(sx);
            int yn = (int)metal::round(sy);
            if (pad_mode == 0) {
                if (xn < 0 || xn >= W || yn < 0 || yn >= H) {
                    out[out_idx] = (T)cval;
                } else {
                    int off = base_idx + yn * h_stride + xn * w_stride;
                    out[out_idx] = x[off];
                }
            } else {
                xn = clamp(xn, 0, W - 1);
                yn = clamp(yn, 0, H - 1);
                int off = base_idx + yn * h_stride + xn * w_stride;
                out[out_idx] = x[off];
            }
            return;
        }

        int x0 = (int)metal::floor(sx);
        int y0 = (int)metal::floor(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float wx = sx - (float)x0;
        float wy = sy - (float)y0;

        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = wx * (1.0f - wy);
        float w10 = (1.0f - wx) * wy;
        float w11 = wx * wy;

        int x0c = clamp(x0, 0, W - 1);
        int y0c = clamp(y0, 0, H - 1);
        int x1c = clamp(x1, 0, W - 1);
        int y1c = clamp(y1, 0, H - 1);

        int off00 = base_idx + y0c * h_stride + x0c * w_stride;
        int off01 = base_idx + y0c * h_stride + x1c * w_stride;
        int off10 = base_idx + y1c * h_stride + x0c * w_stride;
        int off11 = base_idx + y1c * h_stride + x1c * w_stride;

        bool in00 = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H);
        bool in01 = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H);
        bool in10 = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H);
        bool in11 = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H);

        float v00 = (pad_mode == 0 && !in00) ? cval : (float)x[off00];
        float v01 = (pad_mode == 0 && !in01) ? cval : (float)x[off01];
        float v10 = (pad_mode == 0 && !in10) ? cval : (float)x[off10];
        float v11 = (pad_mode == 0 && !in11) ? cval : (float)x[off11];

        float outv = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        out[out_idx] = (T)outv;
    """

    return cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name="imgaug2_warp_affine_bilinear_reflect101_channel",
            input_names=["x", "M", "out_hw", "pad_mode_i", "interp_mode_i", "cval_f"],
            output_names=["out"],
            source=source,
            header=_METAL_HEADER_REFLECT101,
            ensure_row_contiguous=True,
        ),
    )


def warp_affine(
    x_nhwc: MxArray,
    inv_m_flat6: MxArray,
    *,
    out_h: int,
    out_w: int,
    pad_mode: int,
    interp_mode: int,
    cval: float,
) -> MxArray:
    """Warp NHWC images with an affine transform using Metal kernels."""
    _require_metal()
    if x_nhwc.ndim != 4:
        raise ValueError(f"Expected x as (N,H,W,C), got {tuple(x_nhwc.shape)}")
    n = int(x_nhwc.shape[0])
    c = int(x_nhwc.shape[3])
    out_shape = (n, int(out_h), int(out_w), c)

    hw = _out_hw_mx(int(out_h), int(out_w))
    use_channel = (int(out_h) * int(out_w)) <= _WARP_CHANNEL_PARALLEL_THRESHOLD
    kernel = _warp_affine_kernel_channel() if use_channel else _warp_affine_kernel_pixel()

    grid_threads = n * int(out_h) * int(out_w) * c if use_channel else n * int(out_h) * int(out_w)
    pad_mode_i = mx.array(np.array([int(pad_mode)], dtype=np.int32))
    interp_mode_i = mx.array(np.array([int(interp_mode)], dtype=np.int32))
    cval_f = mx.array(np.array([float(cval)], dtype=np.float32))
    out = kernel(
        inputs=[x_nhwc, inv_m_flat6, hw, pad_mode_i, interp_mode_i, cval_f],
        template=[("T", x_nhwc.dtype)],
        grid=(grid_threads, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[x_nhwc.dtype],
    )[0]
    return out


@lru_cache(maxsize=4)
def _warp_perspective_kernel_pixel() -> _MetalKernel:
    _require_metal()

    source = r"""
        uint tid = thread_position_in_grid.x;

        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];

        int outH = out_hw[0];
        int outW = out_hw[1];

        int outHW = outH * outW;
        int b = (int)(tid / (uint)outHW);
        int t = (int)(tid - (uint)(b * outHW));
        int oy = t / outW;
        int ox = t - oy * outW;

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        int out_w_stride = C;
        int out_h_stride = outW * out_w_stride;
        int out_b_stride = outH * out_h_stride;

        int out_base = b * out_b_stride + oy * out_h_stride + ox * out_w_stride;

        float m00 = M[0]; float m01 = M[1]; float m02 = M[2];
        float m10 = M[3]; float m11 = M[4]; float m12 = M[5];
        float m20 = M[6]; float m21 = M[7]; float m22 = M[8];

        float fx = (float)ox;
        float fy = (float)oy;

        float den = m20 * fx + m21 * fy + m22;
        if (den == 0.0f) den = 1.0f;

        float sx = (m00 * fx + m01 * fy + m02) / den;
        float sy = (m10 * fx + m11 * fy + m12) / den;

        int pad_mode = pad_mode_i[0];
        int interp_mode = interp_mode_i[0];
        float cval = cval_f[0];

        if (pad_mode == 2) {
            sx = reflect101_f(sx, W);
            sy = reflect101_f(sy, H);
        } else if (pad_mode == 1) {
            sx = clamp(sx, 0.0f, (float)(W - 1));
            sy = clamp(sy, 0.0f, (float)(H - 1));
        }

        int base_b = b * b_stride;

        if (interp_mode == 0) {
            int xn = (int)metal::round(sx);
            int yn = (int)metal::round(sy);
            if (pad_mode == 0) {
                if (xn < 0 || xn >= W || yn < 0 || yn >= H) {
                    for (int c = 0; c < C; ++c) {
                        out[out_base + c] = (T)cval;
                    }
                } else {
                    int off = base_b + yn * h_stride + xn * w_stride;
                    for (int c = 0; c < C; ++c) {
                        out[out_base + c] = x[off + c];
                    }
                }
            } else {
                xn = clamp(xn, 0, W - 1);
                yn = clamp(yn, 0, H - 1);
                int off = base_b + yn * h_stride + xn * w_stride;
                for (int c = 0; c < C; ++c) {
                    out[out_base + c] = x[off + c];
                }
            }
            return;
        }

        int x0 = (int)metal::floor(sx);
        int y0 = (int)metal::floor(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float wx = sx - (float)x0;
        float wy = sy - (float)y0;

        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = wx * (1.0f - wy);
        float w10 = (1.0f - wx) * wy;
        float w11 = wx * wy;

        int x0c = clamp(x0, 0, W - 1);
        int y0c = clamp(y0, 0, H - 1);
        int x1c = clamp(x1, 0, W - 1);
        int y1c = clamp(y1, 0, H - 1);

        int off00 = base_b + y0c * h_stride + x0c * w_stride;
        int off01 = base_b + y0c * h_stride + x1c * w_stride;
        int off10 = base_b + y1c * h_stride + x0c * w_stride;
        int off11 = base_b + y1c * h_stride + x1c * w_stride;

        bool in00 = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H);
        bool in01 = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H);
        bool in10 = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H);
        bool in11 = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H);

        for (int c = 0; c < C; ++c) {
            float v00 = (pad_mode == 0 && !in00) ? cval : (float)x[off00 + c];
            float v01 = (pad_mode == 0 && !in01) ? cval : (float)x[off01 + c];
            float v10 = (pad_mode == 0 && !in10) ? cval : (float)x[off10 + c];
            float v11 = (pad_mode == 0 && !in11) ? cval : (float)x[off11 + c];
            float outv = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
            out[out_base + c] = (T)outv;
        }
    """

    return cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name="imgaug2_warp_perspective_bilinear_reflect101_pixel",
            input_names=["x", "M", "out_hw", "pad_mode_i", "interp_mode_i", "cval_f"],
            output_names=["out"],
            source=source,
            header=_METAL_HEADER_REFLECT101,
            ensure_row_contiguous=True,
        ),
    )


@lru_cache(maxsize=4)
def _warp_perspective_kernel_channel() -> _MetalKernel:
    _require_metal()

    source = r"""
        uint elem = thread_position_in_grid.x;

        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];

        int outH = out_hw[0];
        int outW = out_hw[1];

        int outHW = outH * outW;

        int pix = (int)(elem / (uint)C);
        int c = (int)(elem - (uint)(pix * C));
        int b = pix / outHW;
        int t = pix - b * outHW;
        int oy = t / outW;
        int ox = t - oy * outW;

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        int base_idx = b * b_stride + c;
        int out_idx = (b * outHW + t) * C + c;

        float m00 = M[0]; float m01 = M[1]; float m02 = M[2];
        float m10 = M[3]; float m11 = M[4]; float m12 = M[5];
        float m20 = M[6]; float m21 = M[7]; float m22 = M[8];

        float fx = (float)ox;
        float fy = (float)oy;

        float den = m20 * fx + m21 * fy + m22;
        if (den == 0.0f) den = 1.0f;

        float sx = (m00 * fx + m01 * fy + m02) / den;
        float sy = (m10 * fx + m11 * fy + m12) / den;

        int pad_mode = pad_mode_i[0];
        int interp_mode = interp_mode_i[0];
        float cval = cval_f[0];

        if (pad_mode == 2) {
            sx = reflect101_f(sx, W);
            sy = reflect101_f(sy, H);
        } else if (pad_mode == 1) {
            sx = clamp(sx, 0.0f, (float)(W - 1));
            sy = clamp(sy, 0.0f, (float)(H - 1));
        }

        if (interp_mode == 0) {
            int xn = (int)metal::round(sx);
            int yn = (int)metal::round(sy);
            if (pad_mode == 0) {
                if (xn < 0 || xn >= W || yn < 0 || yn >= H) {
                    out[out_idx] = (T)cval;
                } else {
                    int off = base_idx + yn * h_stride + xn * w_stride;
                    out[out_idx] = x[off];
                }
            } else {
                xn = clamp(xn, 0, W - 1);
                yn = clamp(yn, 0, H - 1);
                int off = base_idx + yn * h_stride + xn * w_stride;
                out[out_idx] = x[off];
            }
            return;
        }

        int x0 = (int)metal::floor(sx);
        int y0 = (int)metal::floor(sy);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        float wx = sx - (float)x0;
        float wy = sy - (float)y0;

        float w00 = (1.0f - wx) * (1.0f - wy);
        float w01 = wx * (1.0f - wy);
        float w10 = (1.0f - wx) * wy;
        float w11 = wx * wy;

        int x0c = clamp(x0, 0, W - 1);
        int y0c = clamp(y0, 0, H - 1);
        int x1c = clamp(x1, 0, W - 1);
        int y1c = clamp(y1, 0, H - 1);

        int off00 = base_idx + y0c * h_stride + x0c * w_stride;
        int off01 = base_idx + y0c * h_stride + x1c * w_stride;
        int off10 = base_idx + y1c * h_stride + x0c * w_stride;
        int off11 = base_idx + y1c * h_stride + x1c * w_stride;

        bool in00 = (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H);
        bool in01 = (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H);
        bool in10 = (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H);
        bool in11 = (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H);

        float v00 = (pad_mode == 0 && !in00) ? cval : (float)x[off00];
        float v01 = (pad_mode == 0 && !in01) ? cval : (float)x[off01];
        float v10 = (pad_mode == 0 && !in10) ? cval : (float)x[off10];
        float v11 = (pad_mode == 0 && !in11) ? cval : (float)x[off11];

        float outv = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        out[out_idx] = (T)outv;
    """

    return cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name="imgaug2_warp_perspective_bilinear_reflect101_channel",
            input_names=["x", "M", "out_hw", "pad_mode_i", "interp_mode_i", "cval_f"],
            output_names=["out"],
            source=source,
            header=_METAL_HEADER_REFLECT101,
            ensure_row_contiguous=True,
        ),
    )


def warp_perspective(
    x_nhwc: MxArray,
    inv_m_flat9: MxArray,
    *,
    out_h: int,
    out_w: int,
    pad_mode: int,
    interp_mode: int,
    cval: float,
) -> MxArray:
    """Warp NHWC images with a perspective transform using Metal kernels."""
    _require_metal()
    if x_nhwc.ndim != 4:
        raise ValueError(f"Expected x as (N,H,W,C), got {tuple(x_nhwc.shape)}")
    n = int(x_nhwc.shape[0])
    c = int(x_nhwc.shape[3])
    out_shape = (n, int(out_h), int(out_w), c)

    hw = _out_hw_mx(int(out_h), int(out_w))
    use_channel = (int(out_h) * int(out_w)) <= _WARP_CHANNEL_PARALLEL_THRESHOLD
    kernel = _warp_perspective_kernel_channel() if use_channel else _warp_perspective_kernel_pixel()

    grid_threads = n * int(out_h) * int(out_w) * c if use_channel else n * int(out_h) * int(out_w)
    pad_mode_i = mx.array(np.array([int(pad_mode)], dtype=np.int32))
    interp_mode_i = mx.array(np.array([int(interp_mode)], dtype=np.int32))
    cval_f = mx.array(np.array([float(cval)], dtype=np.float32))
    out = kernel(
        inputs=[x_nhwc, inv_m_flat9, hw, pad_mode_i, interp_mode_i, cval_f],
        template=[("T", x_nhwc.dtype)],
        grid=(grid_threads, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[x_nhwc.dtype],
    )[0]
    return out


@lru_cache(maxsize=64)
def _gaussian_blur_sep_tg32_kernel(ksize: int) -> _MetalKernel:
    _require_metal()
    k = int(ksize)
    if k <= 0 or (k % 2) == 0:
        raise ValueError(f"ksize must be positive odd, got {ksize}")

    r = k // 2
    tile = _tile_for_ksize_tg32(k)
    if tile < 8:
        raise ValueError(f"ksize too large for tg32 tiled kernel: ksize={k}, tile={tile}")

    tile_in_sz = 32 * 32
    tile_h_sz = tile * 32

    source = f"""
        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];

        int z = (int)thread_position_in_grid.z;
        int b = z / C;
        int c = z - b * C;

        int tgx = (int)thread_position_in_threadgroup.x;
        int tgy = (int)thread_position_in_threadgroup.y;
        int group_x = (int)threadgroup_position_in_grid.x;
        int group_y = (int)threadgroup_position_in_grid.y;

        constexpr int K = {k};
        constexpr int R = {r};
        constexpr int TILE = {tile};
        constexpr int TGW = 32;
        constexpr int TGH = 32;

        int base_x = group_x * TILE;
        int base_y = group_y * TILE;

        int in_x = base_x + (tgx - R);
        int in_y = base_y + (tgy - R);

        int ix = reflect101_i_mod(in_x, W);
        int iy = reflect101_i_mod(in_y, H);

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        int base = b * b_stride + c;

        threadgroup float tile_in[{tile_in_sz}];
        threadgroup float tile_h[{tile_h_sz}];

        tile_in[tgy * TGW + tgx] = (float)x[base + iy * h_stride + ix * w_stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tgx >= R && tgx < (R + TILE)) {{
            float sum = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < K; ++kk) {{
                float wt = (float)w[kk];
                float v = tile_in[tgy * TGW + (tgx - R + kk)];
                sum = metal::fma(wt, v, sum);
            }}
            tile_h[tgy * TILE + (tgx - R)] = sum;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tgx >= R && tgx < (R + TILE) && tgy >= R && tgy < (R + TILE)) {{
            int ox = base_x + (tgx - R);
            int oy = base_y + (tgy - R);

            if (ox < W && oy < H) {{
                float acc = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < K; ++kk) {{
                    float wt = (float)w[kk];
                    float v = tile_h[(tgy - R + kk) * TILE + (tgx - R)];
                    acc = metal::fma(wt, v, acc);
                }}
                out[base + oy * h_stride + ox * w_stride] = (T)acc;
            }}
        }}
    """

    return cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name=f"imgaug2_gaussian_sep_reflect101_tg32_k{ksize}",
            input_names=["x", "w"],
            output_names=["out"],
            source=source,
            header=_METAL_HEADER_REFLECT101,
            ensure_row_contiguous=True,
        ),
    )


def gaussian_blur_sep_reflect101_tg32(
    x_nhwc: mx.array,
    w_1d: mx.array,
    *,
    ksize: int,
) -> mx.array:
    """Separable Gaussian blur (reflect101) optimized for tg32 tiles."""
    _require_metal()
    if x_nhwc.ndim != 4:
        raise ValueError(f"Expected x as (N,H,W,C), got {tuple(x_nhwc.shape)}")
    if w_1d.ndim != 1 or int(w_1d.shape[0]) != int(ksize):
        raise ValueError(f"Expected w_1d shape ({ksize},), got {tuple(w_1d.shape)}")

    n, h, w, c = map(int, x_nhwc.shape)
    k = int(ksize)
    tile = _tile_for_ksize_tg32(k)
    if tile < 8:
        raise ValueError(f"ksize too large for tg32 tiled kernel: ksize={k}, tile={tile}")

    groups_x = _ceil_div(w, tile)
    groups_y = _ceil_div(h, tile)

    grid = (groups_x * 32, groups_y * 32, n * c)
    threadgroup = (32, 32, 1)

    kernel = _gaussian_blur_sep_tg32_kernel(k)
    out = kernel(
        inputs=[x_nhwc, w_1d],
        template=[("T", x_nhwc.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x_nhwc.shape],
        output_dtypes=[x_nhwc.dtype],
    )[0]
    return out


@lru_cache(maxsize=32)
def _gaussian_blur2d_reflect101_kernel(ksize: int) -> _MetalKernel:
    _require_metal()
    k = int(ksize)
    if k <= 0 or (k % 2) == 0:
        raise ValueError(f"ksize must be positive odd, got {ksize}")
    r = k // 2

    source = f"""
        uint elem = thread_position_in_grid.x;

        int H = x_shape[1];
        int W = x_shape[2];
        int C = x_shape[3];

        int HW = H * W;
        int CHW = C * HW;

        int b = (int)(elem / (uint)CHW);
        int rem = (int)(elem - (uint)(b * CHW));
        int c = rem % C;
        int t = rem / C;
        int oy = t / W;
        int ox = t - oy * W;

        int w_stride = C;
        int h_stride = W * w_stride;
        int b_stride = H * h_stride;

        int base_b = b * b_stride;
        const device T* xptr = x + base_b + c;
        device T* outptr = out + base_b + c;

        int out_loc = oy * h_stride + ox * w_stride;

        constexpr int K = {k};
        constexpr int R = {r};

        float acc = 0.0f;

        bool interior = (ox >= R) && (ox < (W - R)) && (oy >= R) && (oy < (H - R));

        if (interior) {{
            #pragma unroll
            for (int ky = 0; ky < K; ++ky) {{
                int iy = oy + ky - R;
                int yoff = iy * h_stride;
                #pragma unroll
                for (int kx = 0; kx < K; ++kx) {{
                    int ix = ox + kx - R;
                    float wt = w[ky * K + kx];
                    float val = (float)xptr[yoff + ix * w_stride];
                    acc = metal::fma(wt, val, acc);
                }}
            }}
        }} else {{
            #pragma unroll
            for (int ky = 0; ky < K; ++ky) {{
                int iy = reflect101_i_mod(oy + ky - R, H);
                int yoff = iy * h_stride;
                #pragma unroll
                for (int kx = 0; kx < K; ++kx) {{
                    int ix = reflect101_i_mod(ox + kx - R, W);
                    float wt = w[ky * K + kx];
                    float val = (float)xptr[yoff + ix * w_stride];
                    acc = metal::fma(wt, val, acc);
                }}
            }}
        }}

        outptr[out_loc] = (T)acc;
    """

    return cast(
        _MetalKernel,
        mx.fast.metal_kernel(
            name=f"imgaug2_gaussian_blur2d_reflect101_k{ksize}",
            input_names=["x", "w"],
            output_names=["out"],
            source=source,
            header=_METAL_HEADER_REFLECT101,
            ensure_row_contiguous=True,
        ),
    )


@lru_cache(maxsize=256)
def gaussian_kernel_2d_weights_flat_mx(sigma_q: float, ksize: int) -> mx.array:
    """Return flattened 2D Gaussian kernel weights as an MLX array."""
    _require_metal()
    k = int(ksize)
    r = k // 2
    x = np.arange(-r, r + 1, dtype=np.float32)
    k1 = np.exp(-(x * x) / (2.0 * (sigma_q * sigma_q))).astype(np.float32)
    k1 /= float(k1.sum())
    k2 = (k1[:, None] * k1[None, :]).astype(np.float32)
    return mx.array(k2.reshape(-1))


def gaussian_blur2d_reflect101(
    x_nhwc: mx.array,
    w_flat: mx.array,
    *,
    ksize: int,
) -> mx.array:
    """Apply 2D Gaussian blur with reflect101 padding using Metal kernels."""
    _require_metal()
    if x_nhwc.ndim != 4:
        raise ValueError(f"Expected x as (N,H,W,C), got {tuple(x_nhwc.shape)}")
    k = int(ksize)
    kernel = _gaussian_blur2d_reflect101_kernel(k)

    n = int(x_nhwc.shape[0])
    h = int(x_nhwc.shape[1])
    w = int(x_nhwc.shape[2])

    grid_threads = n * h * w * int(x_nhwc.shape[3])
    out = kernel(
        inputs=[x_nhwc, w_flat],
        template=[("T", x_nhwc.dtype)],
        grid=(grid_threads, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x_nhwc.shape],
        output_dtypes=[x_nhwc.dtype],
    )[0]
    return out
