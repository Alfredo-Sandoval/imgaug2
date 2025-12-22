"""Affine-related geometry ops and augmenters."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage import transform as tf

import imgaug2.dtypes as iadt
import imgaug2.imgaug as ia
import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmentables.batches import _BatchInAugmentation
from imgaug2.augmenters._typing import Array, Images, ParamInput, RNGInput
from imgaug2.compat.markers import legacy
from imgaug2.imgaug import _normalize_cv2_input_arr_

from .. import meta
from ._utils import _handle_mode_arg, _handle_order_arg

if TYPE_CHECKING:
    import mlx.core as _mx

    MlxArray: TypeAlias = _mx.array
else:
    MlxArray: TypeAlias = object

ArrayOrMlx: TypeAlias = Array | MlxArray

Shape: TypeAlias = tuple[int, ...]
Shape2D: TypeAlias = tuple[int, int]
Shape3D: TypeAlias = tuple[int, int, int]
Backend: TypeAlias = Literal["auto", "skimage", "cv2"]
Matrix: TypeAlias = NDArray[np.floating[Any]]
Coords: TypeAlias = NDArray[np.floating[Any]]

ScaleInput: TypeAlias = ParamInput | dict[str, ParamInput]
TranslatePercentInput: TypeAlias = ParamInput | dict[str, ParamInput]
TranslatePxComponent: TypeAlias = int | tuple[int, int] | list[int] | iap.StochasticParameter
TranslatePxInput: TypeAlias = TranslatePxComponent | dict[str, TranslatePxComponent]
ShearInput: TypeAlias = ParamInput | dict[str, ParamInput]

_WARP_AFF_VALID_DTYPES_CV2_ORDER_0 = iadt._convert_dtype_strs_to_types(
    "uint8 uint16 int8 int16 int32 float16 float32 float64 bool"
)
_WARP_AFF_VALID_DTYPES_CV2_ORDER_NOT_0 = iadt._convert_dtype_strs_to_types(
    "uint8 uint16 int8 int16 float16 float32 float64 bool"
)

# skimage | cv2
# 0       | cv2.INTER_NEAREST
# 1       | cv2.INTER_LINEAR
# 2       | -
# 3       | cv2.INTER_CUBIC
# 4       | -
_AFFINE_INTERPOLATION_ORDER_SKIMAGE_TO_CV2 = {
    0: cv2.INTER_NEAREST,
    1: cv2.INTER_LINEAR,
    2: cv2.INTER_CUBIC,
    3: cv2.INTER_CUBIC,
    4: cv2.INTER_CUBIC,
}

# constant, edge, symmetric, reflect, wrap
# skimage   | cv2
# constant  | cv2.BORDER_CONSTANT
# edge      | cv2.BORDER_REPLICATE
# symmetric | cv2.BORDER_REFLECT
# reflect   | cv2.BORDER_REFLECT_101
# wrap      | cv2.BORDER_WRAP
_AFFINE_MODE_SKIMAGE_TO_CV2 = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "symmetric": cv2.BORDER_REFLECT,
    "reflect": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP,
}

_PI = 3.141592653589793
_RAD_PER_DEGREE = _PI / 180
def _warp_affine_arr(
    arr: ArrayOrMlx,
    matrix: Matrix,
    order: int = 1,
    mode: str = "constant",
    cval: float | int | Sequence[float | int] | Array = 0,
    output_shape: Shape2D | Shape3D | None = None,
    backend: str = "auto",
) -> ArrayOrMlx:
    # no changes to zero-sized arrays
    if arr.size == 0:
        return arr

    # MLX fast-path (B1): only when input is already on device.
    from imgaug2.mlx._core import is_mlx_array

    if is_mlx_array(arr):
        import imgaug2.mlx as mlx

        order_i = int(order)
        if order_i not in (0, 1):
            raise NotImplementedError(
                "MLX affine warp only supports interpolation orders 0 (nearest) and 1 (bilinear). "
                f"Got order={order_i}."
            )

        output_shape_hw = None
        if output_shape is not None:
            output_shape_hw = (int(output_shape[0]), int(output_shape[1]))

        if ia.is_single_integer(cval) or ia.is_single_float(cval):
            cval_mlx = float(cval)
        else:
            cval_arr = np.array(cval).reshape(-1)
            nb_channels = 1 if arr.ndim == 2 else int(arr.shape[-1])
            if cval_arr.size == 0:
                cval_mlx = 0.0
            elif cval_arr.size == 1:
                cval_mlx = float(cval_arr[0])
            else:
                cval_mlx = cval_arr[:nb_channels].astype(np.float32, copy=False).tolist()

        warped = mlx.affine_transform(
            arr,
            matrix,
            output_shape=output_shape_hw,
            order=order_i,
            cval=cval_mlx,
            mode=mode,
        )
        return warped

    if ia.is_single_integer(cval) or ia.is_single_float(cval):
        cval = [cval] * len(arr.shape[2])

    min_value, _center_value, max_value = iadt.get_value_range_of_dtype(arr.dtype)

    cv2_bad_order = order not in [0, 1, 3]
    if order == 0:
        cv2_bad_dtype = arr.dtype not in _WARP_AFF_VALID_DTYPES_CV2_ORDER_0
    else:
        cv2_bad_dtype = arr.dtype not in _WARP_AFF_VALID_DTYPES_CV2_ORDER_NOT_0
    cv2_impossible = cv2_bad_order or cv2_bad_dtype
    use_skimage = backend == "skimage" or (backend == "auto" and cv2_impossible)
    if use_skimage:
        # cval contains 3 values as cv2 can handle 3, but
        # skimage only 1
        cval = cval[0]
        # skimage does not clip automatically
        cval = max(min(cval, max_value), min_value)
        image_warped = _warp_affine_arr_skimage(
            arr, matrix, cval=cval, mode=mode, order=order, output_shape=output_shape
        )
    else:
        assert not cv2_bad_dtype, (
            not cv2_bad_dtype,
            f"cv2 backend in Affine got a dtype {arr.dtype}, which it "
            "cannot handle. Try using a different dtype or set "
            "order=0.",
        )
        cval_type = float if arr.dtype.kind == "f" else int
        image_warped = _warp_affine_arr_cv2(
            arr,
            matrix,
            cval=tuple([cval_type(v) for v in cval]),
            mode=mode,
            order=order,
            output_shape=output_shape,
        )
    return image_warped


def _warp_affine_arr_skimage(
    arr: Array,
    matrix: Matrix,
    cval: float | int,
    mode: str,
    order: int,
    output_shape: Shape2D | Shape3D | None,
) -> Array:
    iadt.gate_dtypes_strs(
        {arr.dtype},
        allowed="bool uint8 uint16 uint32 int8 int16 int32 float16 float32 float64",
        disallowed="uint64 int64 float128",
    )

    input_dtype = arr.dtype

    # tf.warp() produces a deprecation warning for bool images with
    # order!=0. We either need to convert them to float or use NN
    # interpolation.
    if input_dtype == iadt._BOOL_DTYPE and order != 0:
        arr = arr.astype(np.float32)
    # scipy.ndimage (used internally by skimage) does not support float16.
    # Convert to float32 for the warp and convert back afterwards.
    elif input_dtype == iadt._FLOAT16_DTYPE:
        arr = arr.astype(np.float32)

    image_warped = tf.warp(
        arr,
        np.linalg.inv(matrix),
        order=order,
        mode=mode,
        cval=cval,
        preserve_range=True,
        output_shape=output_shape,
    )

    # tf.warp changes all dtypes to float64, including uint8
    if input_dtype.kind == "b":
        image_warped = image_warped > 0.5
    else:
        image_warped = iadt.restore_dtypes_(image_warped, input_dtype)

    return image_warped


def _warp_affine_arr_cv2(
    arr: Array,
    matrix: Matrix,
    cval: tuple[float | int, ...],
    mode: str | int,
    order: int,
    output_shape: Shape2D | Shape3D,
) -> Array:
    iadt.gate_dtypes_strs(
        {arr.dtype},
        allowed="bool uint8 uint16 int8 int16 int32 float16 float32 float64",
        disallowed="uint32 uint64 int64 float128",
    )

    if order != 0:
        assert arr.dtype != iadt._INT32_DTYPE, (
            "Affine only supports cv2-based transformations of int32 "
            f"arrays when using order=0, but order was set to {order}."
        )

    input_dtype = arr.dtype
    if input_dtype in {iadt._BOOL_DTYPE, iadt._FLOAT16_DTYPE}:
        arr = arr.astype(np.float32)
    elif input_dtype == iadt._INT8_DTYPE and order != 0:
        arr = arr.astype(np.int16)

    dsize = (int(np.round(output_shape[1])), int(np.round(output_shape[0])))

    # map key X from skimage to cv2 or fall back to key X
    mode = _AFFINE_MODE_SKIMAGE_TO_CV2.get(mode, mode)
    order = _AFFINE_INTERPOLATION_ORDER_SKIMAGE_TO_CV2.get(order, order)

    # TODO this uses always a tuple of 3 values for cval, even if
    #      #chans != 3, works with 1d but what in other cases?
    nb_channels = arr.shape[-1]
    if nb_channels <= 3:
        # TODO this block can also be when order==0 for any nb_channels,
        #      but was deactivated for now, because cval would always
        #      contain 3 values and not nb_channels values
        image_warped = cv2.warpAffine(
            _normalize_cv2_input_arr_(arr),
            matrix[0:2, :],
            dsize=dsize,
            flags=order,
            borderMode=mode,
            borderValue=cval,
        )

        # cv2 warp drops last axis if shape is (H, W, 1)
        if image_warped.ndim == 2:
            image_warped = image_warped[..., np.newaxis]
    else:
        # warp each channel on its own, re-add channel axis, then stack
        # the result from a list of [H, W, 1] to (H, W, C).
        image_warped = [
            cv2.warpAffine(
                _normalize_cv2_input_arr_(arr[:, :, c]),
                matrix[0:2, :],
                dsize=dsize,
                flags=order,
                borderMode=mode,
                borderValue=tuple([cval[0]]),
            )
            for c in range(nb_channels)
        ]
        image_warped = np.stack(image_warped, axis=-1)

    if input_dtype.kind == "b":
        image_warped = image_warped > 0.5
    elif input_dtype in {iadt._INT8_DTYPE, iadt._FLOAT16_DTYPE}:
        image_warped = iadt.restore_dtypes_(image_warped, input_dtype)

    return image_warped


@legacy(version="0.5.0")
def _warp_affine_coords(coords: Coords, matrix: Matrix) -> Coords:
    if len(coords) == 0:
        return coords
    assert coords.shape[1] == 2
    assert matrix.shape == (3, 3)

    # this is the same as in scikit-image, _geometric.py -> _apply_mat()
    x, y = np.transpose(coords)
    src = np.vstack((x, y, np.ones_like(x)))
    dst = np.dot(src.T, matrix.T)

    # below, we will divide by the last dimension of the homogeneous
    # coordinate matrix. In order to avoid division by zero,
    # we replace exact zeros in this column with a very small number.
    dst[dst[:, 2] == 0, 2] = np.finfo(float).eps
    # rescale to homogeneous coordinates
    dst[:, :2] /= dst[:, 2:3]

    return dst[:, :2]


def _compute_affine_warp_output_shape(matrix: Matrix, input_shape: Shape) -> tuple[Matrix, Shape]:
    height, width = input_shape[:2]

    if height == 0 or width == 0:
        return matrix, input_shape

    # determine shape of output image
    corners = np.array([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]])
    corners = _warp_affine_coords(corners, matrix)
    minc = corners[:, 0].min()
    minr = corners[:, 1].min()
    maxc = corners[:, 0].max()
    maxr = corners[:, 1].max()
    out_height = maxr - minr + 1
    out_width = maxc - minc + 1
    if len(input_shape) == 3:
        output_shape = np.ceil((out_height, out_width, input_shape[2]))
    else:
        output_shape = np.ceil((out_height, out_width))
    output_shape = tuple([int(v) for v in output_shape.tolist()])
    # fit output image in new shape
    translation = (-minc, -minr)
    matrix = (
        _AffineMatrixGenerator(matrix).translate(x_px=translation[0], y_px=translation[1]).matrix
    )
    return matrix, output_shape

@legacy(version="0.5.0")
class _AffineMatrixGenerator:
    @legacy(version="0.5.0")
    def __init__(self, matrix: Matrix | None = None) -> None:
        if matrix is None:
            matrix = np.eye(3, dtype=np.float32)
        self.matrix = matrix

    @legacy(version="0.5.0")
    def centerize(self, image_shape: Shape) -> _AffineMatrixGenerator:
        height, width = image_shape[0:2]
        self.translate(-width / 2, -height / 2)
        return self

    @legacy(version="0.5.0")
    def invert_centerize(self, image_shape: Shape) -> _AffineMatrixGenerator:
        height, width = image_shape[0:2]
        self.translate(width / 2, height / 2)
        return self

    @legacy(version="0.5.0")
    def translate(self, x_px: float, y_px: float) -> _AffineMatrixGenerator:
        if x_px < 1e-4 or x_px > 1e-4 or y_px < 1e-4 or x_px > 1e-4:
            matrix = np.array([[1, 0, x_px], [0, 1, y_px], [0, 0, 1]], dtype=np.float32)
            self._mul(matrix)
        return self

    @legacy(version="0.5.0")
    def scale(self, x_frac: float, y_frac: float) -> _AffineMatrixGenerator:
        if x_frac < 1.0 - 1e-4 or x_frac > 1.0 + 1e-4 or y_frac < 1.0 - 1e-4 or y_frac > 1.0 + 1e-4:
            matrix = np.array([[x_frac, 0, 0], [0, y_frac, 0], [0, 0, 1]], dtype=np.float32)
            self._mul(matrix)
        return self

    @legacy(version="0.5.0")
    def rotate(self, rad: float) -> _AffineMatrixGenerator:
        if rad < 1e-4 or rad > 1e-4:
            rad = -rad
            matrix = np.array(
                [[np.cos(rad), np.sin(rad), 0], [-np.sin(rad), np.cos(rad), 0], [0, 0, 1]],
                dtype=np.float32,
            )
            self._mul(matrix)
        return self

    @legacy(version="0.5.0")
    def shear(self, x_rad: float, y_rad: float) -> _AffineMatrixGenerator:
        if x_rad < 1e-4 or x_rad > 1e-4 or y_rad < 1e-4 or y_rad > 1e-4:
            matrix = np.array(
                [[1, np.tanh(-x_rad), 0], [np.tanh(y_rad), 1, 0], [0, 0, 1]], dtype=np.float32
            )
            self._mul(matrix)
        return self

    @legacy(version="0.5.0")
    def _mul(self, matrix: Matrix) -> None:
        self.matrix = np.matmul(matrix, self.matrix)


class _AffineSamplingResult:
    def __init__(
        self,
        scale: tuple[Array, Array] | None = None,
        translate: tuple[Array, Array] | None = None,
        translate_mode: Literal["px", "percent"] = "px",
        rotate: Array | None = None,
        shear: tuple[Array, Array] | None = None,
        cval: Array | None = None,
        mode: Sequence[str] | None = None,
        order: Sequence[int] | None = None,
    ) -> None:
        self.scale = scale
        self.translate = translate
        self.translate_mode = translate_mode
        self.rotate = rotate
        self.shear = shear
        self.cval = cval
        self.mode = mode
        self.order = order

    @legacy(version="0.4.0")
    def get_affine_parameters(
        self, idx: int, arr_shape: Shape, image_shape: Shape
    ) -> dict[str, float]:
        scale_y = self.scale[1][idx]
        scale_x = self.scale[0][idx]

        translate_y = self.translate[1][idx]
        translate_x = self.translate[0][idx]
        assert self.translate_mode in ["px", "percent"], (
            f"Expected 'px' or 'percent', got '{self.translate_mode}'."
        )

        if self.translate_mode == "percent":
            translate_y_px = translate_y * arr_shape[0]
            translate_x_px = translate_x * arr_shape[1]
        else:
            translate_y_px = (translate_y / image_shape[0]) * arr_shape[0]
            translate_x_px = (translate_x / image_shape[1]) * arr_shape[1]

        rotate_deg = self.rotate[idx]
        shear_x_deg = self.shear[0][idx]
        shear_y_deg = self.shear[1][idx]

        rotate_rad = rotate_deg * _RAD_PER_DEGREE
        shear_x_rad = shear_x_deg * _RAD_PER_DEGREE
        shear_y_rad = shear_y_deg * _RAD_PER_DEGREE

        # we add the _deg versions of rotate and shear here for PILAffine,
        # Affine itself only uses *_rad
        return {
            "scale_y": scale_y,
            "scale_x": scale_x,
            "translate_y_px": translate_y_px,
            "translate_x_px": translate_x_px,
            "rotate_rad": rotate_rad,
            "shear_y_rad": shear_y_rad,
            "shear_x_rad": shear_x_rad,
            "rotate_deg": rotate_deg,
            "shear_y_deg": shear_y_deg,
            "shear_x_deg": shear_x_deg,
        }

    # for images we use additional shifts of (0.5, 0.5) as otherwise
    # we get an ugly black border for 90deg rotations
    def to_matrix(
        self,
        idx: int,
        arr_shape: Shape,
        image_shape: Shape,
        fit_output: bool,
        shift_add: tuple[float, float] = (0.5, 0.5),
    ) -> tuple[Matrix, Shape]:
        if 0 in image_shape:
            return np.eye(3, dtype=np.float32), arr_shape

        params = self.get_affine_parameters(idx, arr_shape=arr_shape, image_shape=image_shape)

        matrix_gen = _AffineMatrixGenerator()
        matrix_gen.centerize(arr_shape)
        matrix_gen.translate(x_px=shift_add[1], y_px=shift_add[0])
        matrix_gen.rotate(params["rotate_rad"])
        matrix_gen.scale(x_frac=params["scale_x"], y_frac=params["scale_y"])
        matrix_gen.shear(x_rad=params["shear_x_rad"], y_rad=params["shear_y_rad"])
        matrix_gen.translate(x_px=params["translate_x_px"], y_px=params["translate_y_px"])
        matrix_gen.translate(x_px=-shift_add[1], y_px=-shift_add[0])
        matrix_gen.invert_centerize(arr_shape)

        matrix = matrix_gen.matrix
        if fit_output:
            matrix, arr_shape = _compute_affine_warp_output_shape(matrix, arr_shape)
        return matrix, arr_shape

    @legacy(version="0.4.0")
    def to_matrix_cba(
        self,
        idx: int,
        arr_shape: Shape,
        fit_output: bool,
        shift_add: tuple[float, float] = (0.0, 0.0),
    ) -> tuple[Matrix, Shape]:
        return self.to_matrix(idx, arr_shape, arr_shape, fit_output, shift_add)

    @legacy(version="0.4.0")
    def copy(self) -> _AffineSamplingResult:
        return _AffineSamplingResult(
            scale=self.scale,
            translate=self.translate,
            translate_mode=self.translate_mode,
            rotate=self.rotate,
            shear=self.shear,
            cval=self.cval,
            mode=self.mode,
            order=self.order,
        )


def _is_identity_matrix(matrix: Matrix, eps: float = 1e-4) -> bool:
    identity = np.eye(3, dtype=np.float32)
    # about twice as fast as np.allclose()
    return np.average(np.abs(matrix - identity)) <= eps


class Affine(meta.Augmenter):
    """Augmenter that applies affine transformations to images.

    This wraps OpenCV and Scikit-Image affine transformations, supporting
    translation, rotation, scaling, and shearing.

    Note:
        It is strongly recommended to use matching aspect ratios for segmentation maps
        and heatmaps to ensure alignment with transformed images (e.g., if image is
        (200, 100), maps should be (200, 100) or (100, 50)).

    Supported Dtypes:
        - **Fully Supported**: `uint8`, `uint16`, `int8`, `int16`, `float16`, `float32`, `float64`, `bool`.
        - **Limited Support**: `int32` (skimage: yes, cv2: only order=0).
        - **Not Supported**: `uint32`, `uint64`, `int64`, `float128`.



    Parameters:
        scale: Scaling factor. 1.0 is no change.
            - number: Fixed value for both axes.
            - tuple `(a, b)`: Uniformly sampled from `[a, b]`.
            - list: Randomly sampled from list.
            - `StochasticParameter`: Sampled per image.
            - dict `{'x': ..., 'y': ...}`: Independent scaling for axes.
        translate_percent: Translation as a fraction of image height/width.
            - Similar types as `scale`.
        translate_px: Translation in pixels.
            - Similar types as `scale` but using integers (except for StochasticParameters).
        rotate: Rotation in degrees.
            - number: Fixed rotation.
            - tuple `(a, b)`: Uniformly sampled from `[a, b]`.
            - list: Randomly sampled from list.
            - `StochasticParameter`: Sampled per image.
        shear: Shear in degrees.
            - Similar types as `scale`. Supports dict `{'x': ..., 'y': ...}`.
        order: Interpolation order (0-5).
            - 0: Nearest-neighbor
            - 1: Bi-linear (default)
            - 3: Bi-cubic
            - 4: Bi-quartic (slow)
            - 5: Bi-quintic (slow)
            - `imgaug2.ALL`: Uses [0, 1, 3, 4, 5] (skimage) or [0, 1, 3] (cv2).
        cval: Constant value for padding (0-255). Used when `mode="constant"`.
            - number: Fixed value.
            - tuple `(a, b)`: Uniformly sampled from `[a, b]`.
            - `imgaug2.ALL`: Equivalent to (0, 255).
        mode: Boundary handling mode.
            - "constant", "edge", "symmetric", "reflect", "wrap".
            - `imgaug2.ALL`: Uses all valid modes.
        fit_output: If True, adapts output shape to show full transformed image.
        backend: Framework to use: "auto", "cv2", or "skimage".
    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Zoom in on all images by a factor of 2.
        >>> aug = iaa.Affine(scale=2.0)

        >>> # Translate all images on x- and y-axis by 16 pixels.
        >>> aug = iaa.Affine(translate_px=16)

        >>> # Translate by 10 percent of width/height.
        >>> aug = iaa.Affine(translate_percent=0.1)

        >>> # Rotate all images by 35 degrees.
        >>> aug = iaa.Affine(rotate=35)

        >>> # Shear all images by 15 degrees.
        >>> aug = iaa.Affine(shear=15)

        >>> # Translate by a random value between -16 and 16 pixels.
        >>> aug = iaa.Affine(translate_px=(-16, 16))

        >>> # Independent translation on x and y axes.
        >>> aug = iaa.Affine(translate_px={"x": (-16, 16), "y": (-4, 4)})

        >>> # Scale with random interpolation order (nearest or linear).
        >>> aug = iaa.Affine(scale=2.0, order=[0, 1])

        >>> # Translate and fill new pixels with a random color (0-255).
        >>> aug = iaa.Affine(translate_px=16, cval=(0, 255))

        >>> # Translate and fill with black in 50% of images, 'edge' mode in other 50%.
        >>> aug = iaa.Affine(translate_px=16, mode=["constant", "edge"])

        >>> # Shear only on y-axis.
        >>> aug = iaa.Affine(shear={"y": (-45, 45)})
    """

    def __init__(
        self,
        scale: ScaleInput | None = None,
        translate_percent: TranslatePercentInput | None = None,
        translate_px: TranslatePxInput | None = None,
        rotate: ParamInput | None = None,
        shear: ShearInput | None = None,
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        params = [scale, translate_percent, translate_px, rotate, shear]
        if all([p is None for p in params]):
            scale = {"x": (0.9, 1.1), "y": (0.9, 1.1)}
            translate_percent = {"x": (-0.1, 0.1), "y": (-0.1, 0.1)}
            rotate = (-15, 15)
            shear = {"x": (-10, 10), "y": (-10, 10)}
        else:
            scale = scale if scale is not None else 1.0
            rotate = rotate if rotate is not None else 0.0
            shear = shear if shear is not None else 0.0

        assert backend in ["auto", "skimage", "cv2"], (
            f"Expected 'backend' to be \"auto\", \"skimage\" or \"cv2\", got {backend}."
        )
        self.backend = backend
        self.order = _handle_order_arg(order, backend)
        self.cval = iap.handle_cval_arg(cval)
        self.mode = _handle_mode_arg(mode)
        self.scale = self._handle_scale_arg(scale)
        self.translate = self._handle_translate_arg(translate_px, translate_percent)
        self.rotate = iap.handle_continuous_param(
            rotate, "rotate", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )
        self.shear, self._shear_param_type = self._handle_shear_arg(shear)
        self.fit_output = fit_output

        # Special order, mode and cval parameters for heatmaps and
        # segmentation maps. These may either be None or a fixed value.
        # Stochastic parameters are currently *not* supported.
        # If set to None, the same values as for images will be used.
        # That is really not recommended for the cval parameter.
        #
        # Segmentation map augmentation by default always pads with a
        # constant value of 0 (background class id), and always uses nearest
        # neighbour interpolation. While other pad modes and BG class ids
        # could be used, the interpolation mode has to be NN as any other
        # mode would lead to averaging class ids, which makes no sense to do.
        self._order_heatmaps = 3
        self._order_segmentation_maps = 0
        self._mode_heatmaps = "constant"
        self._mode_segmentation_maps = "constant"
        self._cval_heatmaps = 0
        self._cval_segmentation_maps = 0

    @classmethod
    def _handle_scale_arg(
        cls, scale: ScaleInput
    ) -> iap.StochasticParameter | tuple[iap.StochasticParameter, iap.StochasticParameter]:
        if isinstance(scale, dict):
            assert "x" in scale or "y" in scale, (
                "Expected scale dictionary to contain at least key \"x\" or "
                "key \"y\". Found neither of them."
            )
            x = scale.get("x", 1.0)
            y = scale.get("y", 1.0)
            return (
                iap.handle_continuous_param(
                    x,
                    "scale['x']",
                    value_range=(0 + 1e-4, None),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                ),
                iap.handle_continuous_param(
                    y,
                    "scale['y']",
                    value_range=(0 + 1e-4, None),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                ),
            )
        return iap.handle_continuous_param(
            scale, "scale", value_range=(0 + 1e-4, None), tuple_to_uniform=True, list_to_choice=True
        )

    @classmethod
    def _handle_translate_arg(
        cls,
        translate_px: TranslatePxInput | None,
        translate_percent: TranslatePercentInput | None,
    ) -> tuple[iap.StochasticParameter, iap.StochasticParameter | None, Literal["px", "percent"]]:
        if translate_percent is None and translate_px is None:
            translate_px = 0

        assert translate_percent is None or translate_px is None, (
            "Expected either translate_percent or translate_px to be "
            "provided, but neither of them was."
        )

        if translate_percent is not None:
            # translate by percent
            if isinstance(translate_percent, dict):
                assert "x" in translate_percent or "y" in translate_percent, (
                    "Expected translate_percent dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them."
                )
                x = translate_percent.get("x", 0)
                y = translate_percent.get("y", 0)
                return (
                    iap.handle_continuous_param(
                        x,
                        "translate_percent['x']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                    ),
                    iap.handle_continuous_param(
                        y,
                        "translate_percent['y']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                    ),
                    "percent",
                )
            return (
                iap.handle_continuous_param(
                    translate_percent,
                    "translate_percent",
                    value_range=None,
                    tuple_to_uniform=True,
                    list_to_choice=True,
                ),
                None,
                "percent",
            )
        else:
            # translate by pixels
            if isinstance(translate_px, dict):
                assert "x" in translate_px or "y" in translate_px, (
                    "Expected translate_px dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them."
                )
                x = translate_px.get("x", 0)
                y = translate_px.get("y", 0)
                return (
                    iap.handle_discrete_param(
                        x,
                        "translate_px['x']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                        allow_floats=False,
                    ),
                    iap.handle_discrete_param(
                        y,
                        "translate_px['y']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                        allow_floats=False,
                    ),
                    "px",
                )
            return (
                iap.handle_discrete_param(
                    translate_px,
                    "translate_px",
                    value_range=None,
                    tuple_to_uniform=True,
                    list_to_choice=True,
                    allow_floats=False,
                ),
                None,
                "px",
            )

    @legacy(version="0.4.0")
    @classmethod
    def _handle_shear_arg(
        cls, shear: ShearInput
    ) -> tuple[
        iap.StochasticParameter | tuple[iap.StochasticParameter, iap.StochasticParameter],
        Literal["dict", "single-number", "other"],
    ]:
        if isinstance(shear, dict):
            assert "x" in shear or "y" in shear, (
                "Expected shear dictionary to contain at "
                "least key \"x\" or key \"y\". Found neither of them."
            )
            x = shear.get("x", 0)
            y = shear.get("y", 0)
            return (
                iap.handle_continuous_param(
                    x, "shear['x']", value_range=None, tuple_to_uniform=True, list_to_choice=True
                ),
                iap.handle_continuous_param(
                    y, "shear['y']", value_range=None, tuple_to_uniform=True, list_to_choice=True
                ),
            ), "dict"
        else:
            param_type = "other"
            if ia.is_single_number(shear):
                param_type = "single-number"
            return iap.handle_continuous_param(
                shear, "shear", value_range=None, tuple_to_uniform=True, list_to_choice=True
            ), param_type

    @legacy(version="0.4.0")
    def _augment_batch_(
        self,
        batch: _BatchInAugmentation,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> _BatchInAugmentation:
        samples = self._draw_samples(batch.nb_rows, random_state)

        if batch.images is not None:
            batch.images = self._augment_images_by_samples(batch.images, samples)

        if batch.heatmaps is not None:
            batch.heatmaps = self._augment_maps_by_samples(
                batch.heatmaps,
                samples,
                "arr_0to1",
                self._cval_heatmaps,
                self._mode_heatmaps,
                self._order_heatmaps,
                "float32",
            )

        if batch.segmentation_maps is not None:
            batch.segmentation_maps = self._augment_maps_by_samples(
                batch.segmentation_maps,
                samples,
                "arr",
                self._cval_segmentation_maps,
                self._mode_segmentation_maps,
                self._order_segmentation_maps,
                "int32",
            )

        for augm_name in ["keypoints", "bounding_boxes", "polygons", "line_strings"]:
            augm_value = getattr(batch, augm_name)
            if augm_value is not None:
                for i, cbaoi in enumerate(augm_value):
                    matrix, output_shape = samples.to_matrix_cba(i, cbaoi.shape, self.fit_output)

                    if (
                        not _is_identity_matrix(matrix)
                        and not cbaoi.empty
                        and 0 not in cbaoi.shape[0:2]
                    ):
                        if augm_name == "bounding_boxes":
                            # Ensure that 4 points are used for bbs.
                            # to_keypoints_on_images() does return 4 points,
                            # to_xy_array() does not.
                            kpsoi = cbaoi.to_keypoints_on_image()
                            coords = kpsoi.to_xy_array()
                            coords_aug = tf.matrix_transform(coords, matrix)
                            kpsoi = kpsoi.fill_from_xy_array_(coords_aug)
                            cbaoi = cbaoi.invert_to_keypoints_on_image_(kpsoi)
                        else:
                            coords = cbaoi.to_xy_array()
                            coords_aug = tf.matrix_transform(coords, matrix)
                            cbaoi = cbaoi.fill_from_xy_array_(coords_aug)

                    cbaoi.shape = output_shape
                    augm_value[i] = cbaoi

        return batch

    def _augment_images_by_samples(
        self,
        images: Images,
        samples: _AffineSamplingResult,
        image_shapes: Sequence[Shape] | None = None,
        return_matrices: bool = False,
    ) -> Images | tuple[Images, list[Matrix]]:
        if image_shapes is None:
            image_shapes = [image.shape for image in images]

        input_was_array = ia.is_np_array(images)
        input_dtype = None if not input_was_array else images.dtype
        result = []
        matrices = []
        gen = enumerate(
            zip(images, image_shapes, samples.cval, samples.mode, samples.order, strict=True)
        )
        for i, (image, image_shape, cval, mode, order) in gen:
            matrix, output_shape = samples.to_matrix(i, image.shape, image_shape, self.fit_output)

            image_warped = image
            if not _is_identity_matrix(matrix):
                image_warped = _warp_affine_arr(
                    image,
                    matrix,
                    order=order,
                    mode=mode,
                    cval=cval,
                    output_shape=output_shape,
                    backend=self.backend,
                )

            result.append(image_warped)

            if return_matrices:
                matrices.append(matrix)

        # the shapes can change due to fit_output, then it may not be possible
        # to return an array, even when the input was an array
        if input_was_array:
            nb_shapes = len({image.shape for image in result})
            if nb_shapes == 1:
                result = np.array(result, input_dtype)

        if return_matrices:
            result = (result, matrices)
        return result

    @legacy(version="0.4.0")
    def _augment_maps_by_samples(
        self,
        augmentables: list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage],
        samples: _AffineSamplingResult,
        arr_attr_name: str,
        cval: float | int | None,
        mode: str | None,
        order: int | None,
        cval_dtype: str,
    ) -> list[ia.HeatmapsOnImage] | list[ia.SegmentationMapsOnImage]:
        nb_images = len(augmentables)

        samples = samples.copy()
        if cval is not None:
            samples.cval = np.full((nb_images, 1), cval, dtype=cval_dtype)
        if mode is not None:
            samples.mode = [mode] * nb_images
        if order is not None:
            samples.order = [order] * nb_images

        arrs = [getattr(augmentable, arr_attr_name) for augmentable in augmentables]
        image_shapes = [augmentable.shape for augmentable in augmentables]
        arrs_aug, matrices = self._augment_images_by_samples(
            arrs, samples, image_shapes=image_shapes, return_matrices=True
        )

        gen = zip(augmentables, arrs_aug, matrices, samples.order, strict=True)
        for augmentable_i, arr_aug, matrix, order_i in gen:
            # skip augmented HM/SM arrs for which the images were not
            # augmented due to being zero-sized
            if 0 in augmentable_i.shape:
                continue

            # order=3 matches cubic interpolation and can cause values to go
            # outside of the range [0.0, 1.0] not clear whether 4+ also do that
            # We don't clip here for Segmentation Maps, because for these
            # the value range isn't clearly limited to [0, 1] (and they should
            # also never use order=3 to begin with).
            if order_i >= 3 and isinstance(augmentable_i, ia.HeatmapsOnImage):
                arr_aug = np.clip(arr_aug, 0.0, 1.0, out=arr_aug)

            setattr(augmentable_i, arr_attr_name, arr_aug)
            if self.fit_output:
                _, output_shape_i = _compute_affine_warp_output_shape(matrix, augmentable_i.shape)
            else:
                output_shape_i = augmentable_i.shape
            augmentable_i.shape = output_shape_i
        return augmentables

    def _draw_samples(self, nb_samples: int, random_state: iarandom.RNG) -> _AffineSamplingResult:
        rngs = random_state.duplicate(12)

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=rngs[0]),
                self.scale[1].draw_samples((nb_samples,), random_state=rngs[1]),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,), random_state=rngs[2])
            scale_samples = (scale_samples, scale_samples)

        if self.translate[1] is not None:
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,), random_state=rngs[3]),
                self.translate[1].draw_samples((nb_samples,), random_state=rngs[4]),
            )
        else:
            translate_samples = self.translate[0].draw_samples((nb_samples,), random_state=rngs[5])
            translate_samples = (translate_samples, translate_samples)

        rotate_samples = self.rotate.draw_samples((nb_samples,), random_state=rngs[6])
        if self._shear_param_type == "dict":
            shear_samples = (
                self.shear[0].draw_samples((nb_samples,), random_state=rngs[7]),
                self.shear[1].draw_samples((nb_samples,), random_state=rngs[8]),
            )
        elif self._shear_param_type == "single-number":
            # only shear on the x-axis if a single number was given
            shear_samples = self.shear.draw_samples((nb_samples,), random_state=rngs[7])
            shear_samples = (shear_samples, np.zeros_like(shear_samples))
        else:
            shear_samples = self.shear.draw_samples((nb_samples,), random_state=rngs[7])
            shear_samples = (shear_samples, shear_samples)

        cval_samples = self.cval.draw_samples((nb_samples, 3), random_state=rngs[9])
        mode_samples = self.mode.draw_samples((nb_samples,), random_state=rngs[10])
        order_samples = self.order.draw_samples((nb_samples,), random_state=rngs[11])

        return _AffineSamplingResult(
            scale=scale_samples,
            translate=translate_samples,
            translate_mode=self.translate[2],
            rotate=rotate_samples,
            shear=shear_samples,
            cval=cval_samples,
            mode=mode_samples,
            order=order_samples,
        )

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.scale,
            self.translate,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
            self.backend,
            self.fit_output,
        ]


class ScaleX(Affine):
    """Apply affine scaling on the x-axis to input data.

    Wrapper around `Affine`.

    Supported Dtypes:
        See `Affine`.

    Parameters:
        scale: Scaling factor for x-axis. 1.0 is no change.
            - number, tuple, list, or StochasticParameter.
            - No dict input allowed.
        order: Interpolation order. See `Affine`.
        cval: Padding value. See `Affine`.
        mode: Padding mode. See `Affine`.
        fit_output: See `Affine`.
        backend: See `Affine`.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Scale width to 50-150%
        >>> aug = iaa.ScaleX((0.5, 1.5))
    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        scale: ParamInput = (0.5, 1.5),
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            scale={"x": scale},
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class ScaleY(Affine):
    """Apply affine scaling on the y-axis to input data.

    Wrapper around `Affine`.

    Supported Dtypes:
        See `Affine`.

    Parameters:
        scale: Scaling factor for y-axis. 1.0 is no change.
            - number, tuple, list, or StochasticParameter.
            - No dict input allowed.
        order: Interpolation order. See `Affine`.
        cval: Padding value. See `Affine`.
        mode: Padding mode. See `Affine`.
        fit_output: See `Affine`.
        backend: See `Affine`.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Scale height to 50-150%
        >>> aug = iaa.ScaleY((0.5, 1.5))
    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        scale: ParamInput = (0.5, 1.5),
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            scale={"y": scale},
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


# TODO make Affine more efficient for translation-only transformations
class TranslateX(Affine):
    """Apply affine translation on the x-axis.

    Wrapper around `Affine`.

    Supported Dtypes:
        See `Affine`.

    Parameters:
        percent: Translation as fraction of x-axis size (-1.0 to 1.0).
        px: Translation in pixels.
        order: See `Affine`.
        cval: See `Affine`.
        mode: See `Affine`.
        fit_output: See `Affine`.
        backend: See `Affine`.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Translate x-axis by -20 to 20 pixels
        >>> aug = iaa.TranslateX(px=(-20, 20))
    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        percent: ParamInput | None = None,
        px: TranslatePxComponent | None = None,
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        if percent is None and px is None:
            percent = (-0.25, 0.25)

        super().__init__(
            translate_percent=({"x": percent} if percent is not None else None),
            translate_px=({"x": px} if px is not None else None),
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


# TODO make Affine more efficient for translation-only transformations
class TranslateY(Affine):
    """Apply affine translation on the y-axis.

    Wrapper around `Affine`.

    Supported Dtypes:
        See `Affine`.

    Parameters:
        percent: Translation as fraction of y-axis size (-1.0 to 1.0).
        px: Translation in pixels.
        order: See `Affine`.
        cval: See `Affine`.
        mode: See `Affine`.
        fit_output: See `Affine`.
        backend: See `Affine`.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Translate y-axis by -20 to 20 pixels
        >>> aug = iaa.TranslateY(px=(-20, 20))
    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        percent: ParamInput | None = None,
        px: TranslatePxComponent | None = None,
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        if percent is None and px is None:
            percent = (-0.25, 0.25)

        super().__init__(
            translate_percent=({"y": percent} if percent is not None else None),
            translate_px=({"y": px} if px is not None else None),
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class Rotate(Affine):
    """Apply affine rotation.

    Wrapper around `Affine`.

    Supported Dtypes:
        See `Affine`.

    Parameters:
        rotate: Rotation in degrees.
        order: See `Affine`.
        cval: See `Affine`.
        mode: See `Affine`.
        fit_output: See `Affine`.
        backend: See `Affine`.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Rotate -45 to 45 degrees
        >>> aug = iaa.Rotate((-45, 45))
    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        rotate: ParamInput = (-30, 30),
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            rotate=rotate,
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class ShearX(Affine):
    """Apply affine shear on the x-axis.

    Wrapper around `Affine`.

    Supported Dtypes:
        See `Affine`.

    Parameters:
        shear: Shear in degrees.
        order: See `Affine`.
        cval: See `Affine`.
        mode: See `Affine`.
        fit_output: See `Affine`.
        backend: See `Affine`.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Shear x-axis -20 to 20 degrees
        >>> aug = iaa.ShearX((-20, 20))
    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        shear: ParamInput = (-30, 30),
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            shear={"x": shear},
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class ShearY(Affine):
    """Apply affine shear on the y-axis.

    Wrapper around `Affine`.

    Supported Dtypes:
        See `Affine`.

    Parameters:
        shear: Shear in degrees.
        order: See `Affine`.
        cval: See `Affine`.
        mode: See `Affine`.
        fit_output: See `Affine`.
        backend: See `Affine`.
        seed: See `Augmenter`.
        name: See `Augmenter`.

    Example:
        >>> import imgaug2.augmenters as iaa
        >>> # Shear y-axis -20 to 20 degrees
        >>> aug = iaa.ShearY((-20, 20))
    """

    @legacy(version="0.4.0")
    def __init__(
        self,
        shear: ParamInput = (-30, 30),
        order: int | list[int] | iap.StochasticParameter | Literal["ALL"] = 1,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: str | list[str] | iap.StochasticParameter | Literal["ALL"] = "constant",
        fit_output: bool = False,
        backend: Backend = "auto",
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            shear={"y": shear},
            order=order,
            cval=cval,
            mode=mode,
            fit_output=fit_output,
            backend=backend,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )


class AffineCv2(meta.Augmenter):
    """
    **Deprecated.** Augmenter to apply affine transformations to images using
    cv2 (i.e. opencv) backend.

    .. warning::

        This augmenter is deprecated since 0.4.0.
        Use ``Affine(..., backend='cv2')`` instead.

    Affine transformations
    involve:

        - Translation ("move" image on the x-/y-axis)
        - Rotation
        - Scaling ("zoom" in/out)
        - Shear (move one side of the image, turning a square into a trapezoid)

    All such transformations can create "new" pixels in the image without a
    defined content, e.g. if the image is translated to the left, pixels
    are created on the right.
    A method has to be defined to deal with these pixel values. The
    parameters `cval` and `mode` of this class deal with this.

    Some transformations involve interpolations between several pixels
    of the input image to generate output pixel values. The parameter `order`
    deals with the method of interpolation used for this.

    Deprecated since 0.4.0.

    **Supported dtypes**:

        * ``uint8``: yes; fully tested
        * ``uint16``: ?
        * ``uint32``: ?
        * ``uint64``: ?
        * ``int8``: ?
        * ``int16``: ?
        * ``int32``: ?
        * ``int64``: ?
        * ``float16``: ?
        * ``float32``: ?
        * ``float64``: ?
        * ``float128``: ?
        * ``bool``: ?

    Parameters
    ----------
    scale : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Scaling factor to use, where ``1.0`` denotes \"no change\" and
        ``0.5`` is zoomed out to ``50`` percent of the original size.

            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That value will be
              used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_percent : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter or dict {"x": number/tuple/list/StochasticParameter, "y": number/tuple/list/StochasticParameter}, optional
        Translation as a fraction of the image height/width (x-translation,
        y-translation), where ``0`` denotes "no change" and ``0.5`` denotes
        "half of the axis size".

            * If ``None`` then equivalent to ``0.0`` unless `translate_px` has
              a value other than ``None``.
            * If a single number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]``. That sampled fraction
              value will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    translate_px : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter or dict {"x": int/tuple/list/StochasticParameter, "y": int/tuple/list/StochasticParameter}, optional
        Translation in pixels.

            * If ``None`` then equivalent to ``0`` unless `translate_percent`
              has a value other than ``None``.
            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the discrete interval ``[a..b]``. That number
              will be used identically for both x- and y-axis.
            * If a list, then a random value will be sampled from that list
              per image (again, used for both x- and y-axis).
            * If a ``StochasticParameter``, then from that parameter a value
              will be sampled per image (again, used for both x- and y-axis).
            * If a dictionary, then it is expected to have the keys ``x``
              and/or ``y``. Each of these keys can have the same values as
              described above. Using a dictionary allows to set different
              values for the two axis and sampling will then happen
              *independently* per axis, resulting in samples that differ
              between the axes.

    rotate : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Rotation in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``. Rotation happens around the *center* of the
        image, not the top left corner as in some other frameworks.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and used as the rotation
              value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used to
              sample the rotation value per image.

    shear : number or tuple of number or list of number or imgaug2.parameters.StochasticParameter, optional
        Shear in degrees (**NOT** radians), i.e. expected value range is
        around ``[-360, 360]``.

            * If a number, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a value will be uniformly sampled
              per image from the interval ``[a, b]`` and be used as the
              rotation value.
            * If a list, then a random value will be sampled from that list
              per image.
            * If a ``StochasticParameter``, then this parameter will be used
              to sample the shear value per image.

    order : int or list of int or str or list of str or imaug.ALL or imgaug2.parameters.StochasticParameter, optional
        Interpolation order to use. Allowed are:

            * ``cv2.INTER_NEAREST`` (nearest-neighbor interpolation)
            * ``cv2.INTER_LINEAR`` (bilinear interpolation, used by default)
            * ``cv2.INTER_CUBIC`` (bicubic interpolation over ``4x4`` pixel
                neighborhood)
            * ``cv2.INTER_LANCZOS4``
            * string ``nearest`` (same as ``cv2.INTER_NEAREST``)
            * string ``linear`` (same as ``cv2.INTER_LINEAR``)
            * string ``cubic`` (same as ``cv2.INTER_CUBIC``)
            * string ``lanczos4`` (same as ``cv2.INTER_LANCZOS``)

        ``INTER_NEAREST`` (nearest neighbour interpolation) and
        ``INTER_NEAREST`` (linear interpolation) are the fastest.

            * If a single ``int``, then that order will be used for all images.
            * If a string, then it must be one of: ``nearest``, ``linear``,
              ``cubic``, ``lanczos4``.
            * If an iterable of ``int``/``str``, then for each image a random
              value will be sampled from that iterable (i.e. list of allowed
              order values).
            * If ``imgaug2.ALL``, then equivalant to list ``[cv2.INTER_NEAREST,
              cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]``.
            * If ``StochasticParameter``, then that parameter is queried per
              image to sample the order value to use.

    cval : number or tuple of number or list of number or imaug.ALL or imgaug2.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        (E.g. translating by 1px to the right will create a new 1px-wide
        column of pixels on the left of the image).  The value is only used
        when `mode=constant`. The expected value range is ``[0, 255]`` for
        ``uint8`` images. It may be a float value.

            * If this is a single number, then that value will be used
              (e.g. 0 results in black pixels).
            * If a tuple ``(a, b)``, then three values (for three image
              channels) will be uniformly sampled per image from the
              interval ``[a, b]``.
            * If a list, then a random value will be sampled from that list
              per image.
            * If ``imgaug2.ALL`` then equivalent to tuple ``(0, 255)`.
            * If a ``StochasticParameter``, a new value will be sampled from
              the parameter per image.

    mode : int or str or list of str or list of int or imgaug2.ALL or imgaug2.parameters.StochasticParameter,
           optional
        Method to use when filling in newly created pixels.
        Same meaning as in OpenCV's border mode. Let ``abcdefgh`` be an image's
        content and ``|`` be an image boundary after which new pixels are
        filled in, then the valid modes and their behaviour are the following:

            * ``cv2.BORDER_REPLICATE``: ``aaaaaa|abcdefgh|hhhhhhh``
            * ``cv2.BORDER_REFLECT``: ``fedcba|abcdefgh|hgfedcb``
            * ``cv2.BORDER_REFLECT_101``: ``gfedcb|abcdefgh|gfedcba``
            * ``cv2.BORDER_WRAP``: ``cdefgh|abcdefgh|abcdefg``
            * ``cv2.BORDER_CONSTANT``: ``iiiiii|abcdefgh|iiiiiii``,
               where ``i`` is the defined cval.
            * ``replicate``: Same as ``cv2.BORDER_REPLICATE``.
            * ``reflect``: Same as ``cv2.BORDER_REFLECT``.
            * ``reflect_101``: Same as ``cv2.BORDER_REFLECT_101``.
            * ``wrap``: Same as ``cv2.BORDER_WRAP``.
            * ``constant``: Same as ``cv2.BORDER_CONSTANT``.

        The datatype of the parameter may be:

            * If a single ``int``, then it must be one of the ``cv2.BORDER_*``
              constants.
            * If a single string, then it must be one of: ``replicate``,
              ``reflect``, ``reflect_101``, ``wrap``, ``constant``.
            * If a list of ``int``/``str``, then per image a random mode will
              be picked from that list.
            * If ``imgaug2.ALL``, then a random mode from all possible modes
              will be picked.
            * If ``StochasticParameter``, then the mode will be sampled from
              that parameter per image, i.e. it must return only the above
              mentioned strings.

    seed : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    name : None or str, optional
        See :func:`~imgaug2.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug2.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence, optional
        Old name for parameter `seed`.
        Its usage will not yet cause a deprecation warning,
        but it is still recommended to use `seed` now.
        Outdated since 0.4.0.

    deterministic : bool, optional
        Deprecated since 0.4.0.
        See method ``to_deterministic()`` for an alternative and for
        details about what the "deterministic mode" actually does.

    Examples
    --------
    >>> import imgaug2.augmenters as iaa
    >>> aug = iaa.AffineCv2(scale=2.0)

    Zoom in on all images by a factor of ``2``.

    >>> aug = iaa.AffineCv2(translate_px=16)

    Translate all images on the x- and y-axis by 16 pixels (towards the
    bottom right) and fill up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(translate_percent=0.1)

    Translate all images on the x- and y-axis by ``10`` percent of their
    width/height (towards the bottom right). The pixel values are computed
    per axis based on that axis' size. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(rotate=35)

    Rotate all images by ``35`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(shear=15)

    Shear all images by ``15`` *degrees*. Fill up any new pixels with zero
    (black values).

    >>> aug = iaa.AffineCv2(translate_px=(-16, 16))

    Translate all images on the x- and y-axis by a random value
    between ``-16`` and ``16`` pixels (to the bottom right) and fill up any new
    pixels with zero (black values). The translation value is sampled once
    per image and is the same for both axis.

    >>> aug = iaa.AffineCv2(translate_px={"x": (-16, 16), "y": (-4, 4)})

    Translate all images on the x-axis by a random value
    between ``-16`` and ``16`` pixels (to the right) and on the y-axis by a
    random value between ``-4`` and ``4`` pixels to the bottom. The sampling
    happens independently per axis, so even if both intervals were identical,
    the sampled axis-wise values would likely be different.
    This also fills up any new pixels with zero (black values).

    >>> aug = iaa.AffineCv2(scale=2.0, order=[0, 1])

    Same as in the above `scale` example, but uses (randomly) either
    nearest neighbour interpolation or linear interpolation. If `order` is
    not specified, ``order=1`` would be used by default.

    >>> aug = iaa.AffineCv2(translate_px=16, cval=(0, 255))

    Same as in the `translate_px` example above, but newly created pixels
    are now filled with a random color (sampled once per image and the
    same for all newly created pixels within that image).

    >>> aug = iaa.AffineCv2(translate_px=16, mode=["constant", "replicate"])

    Similar to the previous example, but the newly created pixels are
    filled with black pixels in half of all images (mode ``constant`` with
    default `cval` being ``0``) and in the other half of all images using
    ``replicate`` mode, which repeats the color of the spatially closest pixel
    of the corresponding image edge.

    """

    def __init__(
        self,
        scale: ScaleInput = 1.0,
        translate_percent: TranslatePercentInput | None = None,
        translate_px: TranslatePxInput | None = None,
        rotate: ParamInput = 0.0,
        shear: ParamInput = 0.0,
        order: int
        | str
        | list[int | str]
        | iap.StochasticParameter
        | Literal["ALL"] = cv2.INTER_LINEAR,
        cval: ParamInput | Literal["ALL"] = 0,
        mode: int
        | str
        | list[int | str]
        | iap.StochasticParameter
        | Literal["ALL"] = cv2.BORDER_CONSTANT,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        super().__init__(
            seed=seed, name=name, random_state=random_state, deterministic=deterministic
        )

        # using a context on __init__ seems to produce no warning,
        # so warn manually here
        ia.warn_deprecated(
            "AffineCv2 is deprecated. "
            "Use imgaug2.augmenters.geometric.Affine(..., backend='cv2') "
            "instead.",
            stacklevel=4,
        )

        available_orders = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        available_orders_str = ["nearest", "linear", "cubic", "lanczos4"]

        if order == ia.ALL:
            self.order = iap.Choice(available_orders)
        elif ia.is_single_integer(order):
            assert order in available_orders, (
                f"Expected order's integer value to be in {available_orders}, got {order}."
            )
            self.order = iap.Deterministic(order)
        elif ia.is_string(order):
            assert order in available_orders_str, (
                f"Expected order to be in {str(available_orders_str)}, got {order}."
            )
            self.order = iap.Deterministic(order)
        elif isinstance(order, list):
            valid_types = all([ia.is_single_integer(val) or ia.is_string(val) for val in order])
            assert valid_types, (
                "Expected order list to only contain integers/strings, got "
                f"types {str([type(val) for val in order])}."
            )
            valid_orders = all([val in available_orders + available_orders_str for val in order])
            assert valid_orders, (
                f"Expected all order values to be in {available_orders + available_orders_str}, got {str(order)}."
            )
            self.order = iap.Choice(order)
        elif isinstance(order, iap.StochasticParameter):
            self.order = order
        else:
            raise Exception(
                "Expected order to be imgaug2.ALL, int, string, a list of"
                f"int/string or StochasticParameter, got {type(order)}."
            )

        if cval == ia.ALL:
            self.cval = iap.DiscreteUniform(0, 255)
        else:
            self.cval = iap.handle_discrete_param(
                cval,
                "cval",
                value_range=(0, 255),
                tuple_to_uniform=True,
                list_to_choice=True,
                allow_floats=True,
            )

        available_modes = [
            cv2.BORDER_REPLICATE,
            cv2.BORDER_REFLECT,
            cv2.BORDER_REFLECT_101,
            cv2.BORDER_WRAP,
            cv2.BORDER_CONSTANT,
        ]
        available_modes_str = ["replicate", "reflect", "reflect_101", "wrap", "constant"]
        if mode == ia.ALL:
            self.mode = iap.Choice(available_modes)
        elif ia.is_single_integer(mode):
            assert mode in available_modes, f"Expected mode to be in {available_modes}, got {mode}."
            self.mode = iap.Deterministic(mode)
        elif ia.is_string(mode):
            assert mode in available_modes_str, (
                f"Expected mode to be in {str(available_modes_str)}, got {mode}."
            )
            self.mode = iap.Deterministic(mode)
        elif isinstance(mode, list):
            all_valid_types = all([ia.is_single_integer(val) or ia.is_string(val) for val in mode])
            assert all_valid_types, (
                "Expected mode list to only contain integers/strings, "
                f"got types {str([type(val) for val in mode])}."
            )
            all_valid_modes = all([val in available_modes + available_modes_str for val in mode])
            assert all_valid_modes, (
                f"Expected all mode values to be in {str(available_modes + available_modes_str)}, got {str(mode)}."
            )
            self.mode = iap.Choice(mode)
        elif isinstance(mode, iap.StochasticParameter):
            self.mode = mode
        else:
            raise Exception(
                "Expected mode to be imgaug2.ALL, an int, a string, a list of "
                f"int/strings or StochasticParameter, got {type(mode)}."
            )

        # scale
        if isinstance(scale, dict):
            assert "x" in scale or "y" in scale, (
                "Expected scale dictionary to contain at "
                "least key \"x\" or key \"y\". Found neither of them."
            )
            x = scale.get("x", 1.0)
            y = scale.get("y", 1.0)
            self.scale = (
                iap.handle_continuous_param(
                    x,
                    "scale['x']",
                    value_range=(0 + 1e-4, None),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                ),
                iap.handle_continuous_param(
                    y,
                    "scale['y']",
                    value_range=(0 + 1e-4, None),
                    tuple_to_uniform=True,
                    list_to_choice=True,
                ),
            )
        else:
            self.scale = iap.handle_continuous_param(
                scale,
                "scale",
                value_range=(0 + 1e-4, None),
                tuple_to_uniform=True,
                list_to_choice=True,
            )

        # translate
        if translate_percent is None and translate_px is None:
            translate_px = 0

        assert translate_percent is None or translate_px is None, (
            "Expected either translate_percent or translate_px to be "
            "provided, but neither of them was."
        )

        if translate_percent is not None:
            # translate by percent
            if isinstance(translate_percent, dict):
                assert "x" in translate_percent or "y" in translate_percent, (
                    "Expected translate_percent dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them."
                )
                x = translate_percent.get("x", 0)
                y = translate_percent.get("y", 0)
                self.translate = (
                    iap.handle_continuous_param(
                        x,
                        "translate_percent['x']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                    ),
                    iap.handle_continuous_param(
                        y,
                        "translate_percent['y']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                    ),
                )
            else:
                self.translate = iap.handle_continuous_param(
                    translate_percent,
                    "translate_percent",
                    value_range=None,
                    tuple_to_uniform=True,
                    list_to_choice=True,
                )
        else:
            # translate by pixels
            if isinstance(translate_px, dict):
                assert "x" in translate_px or "y" in translate_px, (
                    "Expected translate_px dictionary to contain at "
                    "least key \"x\" or key \"y\". Found neither of them."
                )
                x = translate_px.get("x", 0)
                y = translate_px.get("y", 0)
                self.translate = (
                    iap.handle_discrete_param(
                        x,
                        "translate_px['x']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                        allow_floats=False,
                    ),
                    iap.handle_discrete_param(
                        y,
                        "translate_px['y']",
                        value_range=None,
                        tuple_to_uniform=True,
                        list_to_choice=True,
                        allow_floats=False,
                    ),
                )
            else:
                self.translate = iap.handle_discrete_param(
                    translate_px,
                    "translate_px",
                    value_range=None,
                    tuple_to_uniform=True,
                    list_to_choice=True,
                    allow_floats=False,
                )

        self.rotate = iap.handle_continuous_param(
            rotate, "rotate", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )
        self.shear = iap.handle_continuous_param(
            shear, "shear", value_range=None, tuple_to_uniform=True, list_to_choice=True
        )

    def _augment_images(
        self,
        images: Images,
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksImages | None,
    ) -> Images:
        nb_images = len(images)
        (
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            cval_samples,
            mode_samples,
            order_samples,
        ) = self._draw_samples(nb_images, random_state)
        result = self._augment_images_by_samples(
            images,
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            cval_samples,
            mode_samples,
            order_samples,
        )
        return result

    @classmethod
    def _augment_images_by_samples(
        cls,
        images: Images,
        scale_samples: tuple[Array, Array],
        translate_samples: tuple[Array, Array],
        rotate_samples: Array,
        shear_samples: Array,
        cval_samples: Array,
        mode_samples: Sequence[int | str],
        order_samples: Sequence[int | str],
    ) -> Images:
        # TODO change these to class attributes
        order_str_to_int = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }
        mode_str_to_int = {
            "replicate": cv2.BORDER_REPLICATE,
            "reflect": cv2.BORDER_REFLECT,
            "reflect_101": cv2.BORDER_REFLECT_101,
            "wrap": cv2.BORDER_WRAP,
            "constant": cv2.BORDER_CONSTANT,
        }

        nb_images = len(images)
        result = images
        for i in range(nb_images):
            height, width = images[i].shape[0], images[i].shape[1]
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x = translate_samples[0][i]
            translate_y = translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(np.round(translate_y * images[i].shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(np.round(translate_x * images[i].shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]
            cval = cval_samples[i]
            mode = mode_samples[i]
            order = order_samples[i]

            mode = mode if ia.is_single_integer(mode) else mode_str_to_int[mode]
            order = order if ia.is_single_integer(order) else order_str_to_int[order]

            any_change = (
                scale_x != 1.0
                or scale_y != 1.0
                or translate_x_px != 0
                or translate_y_px != 0
                or rotate != 0
                or shear != 0
            )

            if any_change:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear),
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = matrix_to_topleft + matrix_transforms + matrix_to_center

                image_warped = cv2.warpAffine(
                    _normalize_cv2_input_arr_(images[i]),
                    matrix.params[:2],
                    dsize=(width, height),
                    flags=order,
                    borderMode=mode,
                    borderValue=tuple([int(v) for v in cval]),
                )

                # cv2 warp drops last axis if shape is (H, W, 1)
                if image_warped.ndim == 2:
                    image_warped = image_warped[..., np.newaxis]

                # warp changes uint8 to float64, making this necessary
                result[i] = image_warped
            else:
                result[i] = images[i]

        return result

    def _augment_heatmaps(
        self,
        heatmaps: list[ia.HeatmapsOnImage],
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksHeatmaps | None,
    ) -> list[ia.HeatmapsOnImage]:
        nb_images = len(heatmaps)
        (
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            cval_samples,
            mode_samples,
            order_samples,
        ) = self._draw_samples(nb_images, random_state)
        cval_samples = np.zeros((cval_samples.shape[0], 1), dtype=np.float32)
        mode_samples = ["constant"] * len(mode_samples)
        arrs = [heatmap_i.arr_0to1 for heatmap_i in heatmaps]
        arrs_aug = self._augment_images_by_samples(
            arrs,
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            cval_samples,
            mode_samples,
            order_samples,
        )
        for heatmap_i, arr_aug in zip(heatmaps, arrs_aug, strict=True):
            heatmap_i.arr_0to1 = arr_aug
        return heatmaps

    def _augment_segmentation_maps(
        self,
        segmaps: list[ia.SegmentationMapsOnImage],
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksSegmentationMaps | None,
    ) -> list[ia.SegmentationMapsOnImage]:
        nb_images = len(segmaps)
        (
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            cval_samples,
            mode_samples,
            order_samples,
        ) = self._draw_samples(nb_images, random_state)
        cval_samples = np.zeros((cval_samples.shape[0], 1), dtype=np.float32)
        mode_samples = ["constant"] * len(mode_samples)
        order_samples = [0] * len(order_samples)
        arrs = [segmaps_i.arr for segmaps_i in segmaps]
        arrs_aug = self._augment_images_by_samples(
            arrs,
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            cval_samples,
            mode_samples,
            order_samples,
        )
        for segmaps_i, arr_aug in zip(segmaps, arrs_aug, strict=True):
            segmaps_i.arr = arr_aug
        return segmaps

    def _augment_keypoints(
        self,
        keypoints_on_images: list[ia.KeypointsOnImage],
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksKeypoints | None,
    ) -> list[ia.KeypointsOnImage]:
        result = []
        nb_images = len(keypoints_on_images)
        (
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            _cval_samples,
            _mode_samples,
            _order_samples,
        ) = self._draw_samples(nb_images, random_state)

        for i, keypoints_on_image in enumerate(keypoints_on_images):
            if not keypoints_on_image.keypoints:
                # AffineCv2 does not change the image shape, hence we can skip
                # all steps below if there are no keypoints
                result.append(keypoints_on_image)
                continue
            height, width = keypoints_on_image.height, keypoints_on_image.width
            shift_x = width / 2.0 - 0.5
            shift_y = height / 2.0 - 0.5
            scale_x, scale_y = scale_samples[0][i], scale_samples[1][i]
            translate_x = translate_samples[0][i]
            translate_y = translate_samples[1][i]
            if ia.is_single_float(translate_y):
                translate_y_px = int(np.round(translate_y * keypoints_on_image.shape[0]))
            else:
                translate_y_px = translate_y
            if ia.is_single_float(translate_x):
                translate_x_px = int(np.round(translate_x * keypoints_on_image.shape[1]))
            else:
                translate_x_px = translate_x
            rotate = rotate_samples[i]
            shear = shear_samples[i]

            any_change = (
                scale_x != 1.0
                or scale_y != 1.0
                or translate_x_px != 0
                or translate_y_px != 0
                or rotate != 0
                or shear != 0
            )

            if any_change:
                matrix_to_topleft = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
                matrix_transforms = tf.AffineTransform(
                    scale=(scale_x, scale_y),
                    translation=(translate_x_px, translate_y_px),
                    rotation=math.radians(rotate),
                    shear=math.radians(shear),
                )
                matrix_to_center = tf.SimilarityTransform(translation=[shift_x, shift_y])
                matrix = matrix_to_topleft + matrix_transforms + matrix_to_center

                coords = keypoints_on_image.to_xy_array()
                coords_aug = tf.matrix_transform(coords, matrix.params)
                kps_new = [
                    kp.deepcopy(x=coords[0], y=coords[1])
                    for kp, coords in zip(keypoints_on_image.keypoints, coords_aug, strict=True)
                ]
                result.append(
                    keypoints_on_image.deepcopy(keypoints=kps_new, shape=keypoints_on_image.shape)
                )
            else:
                result.append(keypoints_on_image)
        return result

    def _augment_polygons(
        self,
        polygons_on_images: list[ia.PolygonsOnImage],
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksPolygons | None,
    ) -> list[ia.PolygonsOnImage]:
        return self._augment_polygons_as_keypoints(polygons_on_images, random_state, parents, hooks)

    def _augment_line_strings(
        self,
        line_strings_on_images: list[ia.LineStringsOnImage],
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksLineStrings | None,
    ) -> list[ia.LineStringsOnImage]:
        return self._augment_line_strings_as_keypoints(
            line_strings_on_images, random_state, parents, hooks
        )

    def _augment_bounding_boxes(
        self,
        bounding_boxes_on_images: list[ia.BoundingBoxesOnImage],
        random_state: iarandom.RNG,
        parents: list[meta.Augmenter],
        hooks: ia.HooksBoundingBoxes | None,
    ) -> list[ia.BoundingBoxesOnImage]:
        return self._augment_bounding_boxes_as_keypoints(
            bounding_boxes_on_images, random_state, parents, hooks
        )

    def get_parameters(self) -> list[object]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        return [
            self.scale,
            self.translate,
            self.rotate,
            self.shear,
            self.order,
            self.cval,
            self.mode,
        ]

    def _draw_samples(
        self, nb_samples: int, random_state: iarandom.RNG
    ) -> tuple[
        tuple[Array, Array],
        tuple[Array, Array],
        Array,
        Array,
        Array,
        Sequence[int | str],
        Sequence[int | str],
    ]:
        rngs = random_state.duplicate(11)

        if isinstance(self.scale, tuple):
            scale_samples = (
                self.scale[0].draw_samples((nb_samples,), random_state=rngs[0]),
                self.scale[1].draw_samples((nb_samples,), random_state=rngs[1]),
            )
        else:
            scale_samples = self.scale.draw_samples((nb_samples,), random_state=rngs[2])
            scale_samples = (scale_samples, scale_samples)

        if isinstance(self.translate, tuple):
            translate_samples = (
                self.translate[0].draw_samples((nb_samples,), random_state=rngs[3]),
                self.translate[1].draw_samples((nb_samples,), random_state=rngs[4]),
            )
        else:
            translate_samples = self.translate.draw_samples((nb_samples,), random_state=rngs[5])
            translate_samples = (translate_samples, translate_samples)

        valid_dts = iadt._convert_dtype_strs_to_types("int32 int64 float32 float64")
        for i in range(2):
            assert translate_samples[i].dtype in valid_dts, (
                f"Expected translate_samples to have any dtype of {str(valid_dts)}. "
                f"Got {translate_samples[i].dtype.name}."
            )

        rotate_samples = self.rotate.draw_samples((nb_samples,), random_state=rngs[6])
        shear_samples = self.shear.draw_samples((nb_samples,), random_state=rngs[7])

        cval_samples = self.cval.draw_samples((nb_samples, 3), random_state=rngs[8])
        mode_samples = self.mode.draw_samples((nb_samples,), random_state=rngs[9])
        order_samples = self.order.draw_samples((nb_samples,), random_state=rngs[10])

        return (
            scale_samples,
            translate_samples,
            rotate_samples,
            shear_samples,
            cval_samples,
            mode_samples,
            order_samples,
        )


__all__ = [
    "Affine",
    "AffineCv2",
    "Rotate",
    "ScaleX",
    "ScaleY",
    "ShearX",
    "ShearY",
    "TranslateX",
    "TranslateY",
]
