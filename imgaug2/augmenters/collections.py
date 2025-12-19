"""Augmenters that are collections of other augmenters.

This module provides preset augmenter pipelines that combine multiple
augmentation techniques for common use cases.

Key Augmenters:
    - `RandAugment`: Implementation of the RandAugment algorithm.
    - `PosePreset`: Preset augmentation pipelines for pose estimation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np

import imgaug2.parameters as iap
import imgaug2.random as iarandom
from imgaug2.augmenters import arithmetic, contrast, flip, geometric, meta, pillike
from imgaug2.augmenters import size as sizelib
from imgaug2.augmenters._typing import Numberish, RNGInput
from imgaug2.compat.markers import legacy

DiscreteParamInput: TypeAlias = int | tuple[int, int] | list[int] | iap.StochasticParameter | None
FillColor: TypeAlias = int | Sequence[int] | iap.StochasticParameter
PosePresetName: TypeAlias = Literal[
    "lightning_pose_dlc",
    "lightning_pose_dlc_lr",
    "lightning_pose_dlc_top_down",
    "deeplabcut_pytorch_default",
    "sleap_default",
    "mmpose_default",
]


@legacy(version="0.4.0")
class RandAugment(meta.Sequential):
    """Apply RandAugment to inputs as described in the corresponding paper.

    See paper::

        Cubuk et al.

        RandAugment: Practical automated data augmentation with a reduced
        search space

    .. note::

        The paper contains essentially no hyperparameters for the individual
        augmentation techniques. The hyperparameters used here come mostly
        from the official code repository, which however seems to only contain
        code for CIFAR10 and SVHN, not for ImageNet. So some guesswork was
        involved and a few of the hyperparameters were also taken from
        https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py .

        This implementation deviates from the code repository for all PIL
        enhance operations. In the repository these use a factor of
        ``0.1 + M*1.8/M_max``, which would lead to a factor of ``0.1`` for the
        weakest ``M`` of ``M=0``. For e.g. ``Brightness`` that would result in
        a basically black image. This definition is fine for AutoAugment (from
        where the code and hyperparameters are copied), which optimizes
        each transformation's ``M`` individually, but not for RandAugment,
        which uses a single fixed ``M``. We hence redefine these
        hyperparameters to ``1.0 + S * M * 0.9/M_max``, where ``S`` is
        randomly either ``1`` or ``-1``.

        We also note that it is not entirely clear which transformations
        were used in the ImageNet experiments. The paper lists some
        transformations in Figure 2, but names others in the text too (e.g.
        crops, flips, cutout). While Figure 2 lists the Identity function,
        this transformation seems to not appear in the repository (and in fact,
        the function ``randaugment(N, M)`` doesn't seem to exist in the
        repository either). So we also make a best guess here about what
        transformations might have been used.

    .. warning::

        This augmenter only works with image data, not e.g. bounding boxes.
        The used PIL-based affine transformations are not yet able to
        process non-image data. (This augmenter uses PIL-based affine
        transformations to ensure that outputs are as similar as possible
        to the paper's implementation.)


    **Supported dtypes**:

    minimum of (
        :class:`~imgaug2.augmenters.flip.Fliplr`,
        :class:`~imgaug2.augmenters.size.KeepSizeByResize`,
        :class:`~imgaug2.augmenters.size.Crop`,
        :class:`~imgaug2.augmenters.meta.Sequential`,
        :class:`~imgaug2.augmenters.meta.SomeOf`,
        :class:`~imgaug2.augmenters.meta.Identity`,
        :class:`~imgaug2.augmenters.pillike.Autocontrast`,
        :class:`~imgaug2.augmenters.pillike.Equalize`,
        :class:`~imgaug2.augmenters.arithmetic.Invert`,
        :class:`~imgaug2.augmenters.pillike.Affine`,
        :class:`~imgaug2.augmenters.pillike.Posterize`,
        :class:`~imgaug2.augmenters.pillike.Solarize`,
        :class:`~imgaug2.augmenters.pillike.EnhanceColor`,
        :class:`~imgaug2.augmenters.pillike.EnhanceContrast`,
        :class:`~imgaug2.augmenters.pillike.EnhanceBrightness`,
        :class:`~imgaug2.augmenters.pillike.EnhanceSharpness`,
        :class:`~imgaug2.augmenters.arithmetic.Cutout`,
        :class:`~imgaug2.augmenters.pillike.FilterBlur`,
        :class:`~imgaug2.augmenters.pillike.FilterSmooth`
    )

    Parameters
    ----------
    n : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter or None, optional
        Parameter ``N`` in the paper, i.e. number of transformations to apply.
        The paper suggests ``N=2`` for ImageNet.
        See also parameter ``n`` in :class:`~imgaug2.augmenters.meta.SomeOf`
        for more details.

        Note that horizontal flips (p=50%) and crops are always applied. This
        parameter only determines how many of the other transformations
        are applied per image.


    m : int or tuple of int or list of int or imgaug2.parameters.StochasticParameter or None, optional
        Parameter ``M`` in the paper, i.e. magnitude/severity/strength of the
        applied transformations in interval ``[0 .. 30]`` with ``M=0`` being
        the weakest. The paper suggests for ImageNet ``M=9`` in case of
        ResNet-50 and ``M=28`` in case of EfficientNet-B7.
        This implementation uses a default value of ``(6, 12)``, i.e. the
        value is uniformly sampled per image from the interval ``[6 .. 12]``.
        This ensures greater diversity of transformations than using a single
        fixed value.

        * If ``int``: That value will always be used.
        * If ``tuple`` ``(a, b)``: A random value will be uniformly sampled per
          image from the discrete interval ``[a .. b]``.
        * If ``list``: A random value will be picked from the list per image.
        * If ``StochasticParameter``: For ``B`` images in a batch, ``B`` values
          will be sampled per augmenter (provided the augmenter is dependent
          on the magnitude).

    cval : number or tuple of number or list of number or imgaug2.ALL or imgaug2.parameters.StochasticParameter, optional
        The constant value to use when filling in newly created pixels.
        See parameter `fillcolor` in
        :class:`~imgaug2.augmenters.pillike.Affine` for details.

        The paper's repository uses an RGB value of ``125, 122, 113``.
        This implementation uses a single intensity value of ``128``, which
        should work better for cases where input images don't have exactly
        ``3`` channels or come from a different dataset than used by the
        paper.

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
    >>> aug = iaa.RandAugment(n=2, m=9)

    Create a RandAugment augmenter similar to the suggested hyperparameters
    in the paper.

    >>> aug = iaa.RandAugment(m=30)

    Create a RandAugment augmenter with maximum magnitude/strength.

    >>> aug = iaa.RandAugment(m=(0, 9))

    Create a RandAugment augmenter that applies its transformations with a
    random magnitude between ``0`` (very weak) and ``9`` (recommended for
    ImageNet and ResNet-50). ``m`` is sampled per transformation.

    >>> aug = iaa.RandAugment(n=(0, 3))

    Create a RandAugment augmenter that applies ``0`` to ``3`` of its
    child transformations to images. Horizontal flips (p=50%) and crops are
    always applied.

    """

    _M_MAX = 30

    # according to paper:
    # N=2, M=9 is optimal for ImageNet with ResNet-50
    # N=2, M=28 is optimal for ImageNet with EfficientNet-B7
    # for cval they use [125, 122, 113]
    @legacy(version="0.4.0")
    def __init__(
        self,
        n: DiscreteParamInput = 2,
        m: DiscreteParamInput = (6, 12),
        cval: FillColor = 128,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        seed = seed if random_state == "deprecated" else random_state
        rng = iarandom.RNG.create_if_not_rng_(seed)

        # we don't limit the value range to 10 here, because the paper
        # gives several examples of using more than 10 for M
        m = iap.handle_discrete_param(
            m,
            "m",
            value_range=(0, None),
            tuple_to_uniform=True,
            list_to_choice=True,
            allow_floats=False,
        )
        self._m = m
        self._cval = cval

        # The paper says in Appendix A.2.3 "ImageNet", that they actually
        # always execute Horizontal Flips and Crops first and only then a
        # random selection of the other transformations.
        # Hence, we split here into two groups.
        # It's not really clear what crop parameters they use, so we
        # choose [0..M] here.
        initial_augs = self._create_initial_augmenters_list(m)
        main_augs = self._create_main_augmenters_list(m, cval)

        # assign random state to all child augmenters
        for lst in [initial_augs, main_augs]:
            for augmenter in lst:
                augmenter.random_state = rng

        super().__init__(
            [
                meta.Sequential(initial_augs, seed=rng.derive_rng_()),
                meta.SomeOf(n, main_augs, random_order=True, seed=rng.derive_rng_()),
            ],
            seed=rng,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )

    @legacy(version="0.4.0")
    @classmethod
    def _create_initial_augmenters_list(cls, m: iap.StochasticParameter) -> list[meta.Augmenter]:
        return [
            flip.Fliplr(0.5),
            sizelib.KeepSizeByResize(
                # assuming that the paper implementation crops M pixels from
                # 224px ImageNet images, we crop here a fraction of
                # M*(M_max/224)
                sizelib.Crop(
                    percent=iap.Divide(iap.Uniform(0, m), 224, elementwise=True),
                    sample_independently=True,
                    keep_size=False,
                ),
                interpolation="linear",
            ),
        ]

    @legacy(version="0.4.0")
    @classmethod
    def _create_main_augmenters_list(
        cls, m: iap.StochasticParameter, cval: FillColor
    ) -> list[meta.Augmenter]:
        m_max = cls._M_MAX

        def _float_parameter(
            level: iap.StochasticParameter, maxval: float
        ) -> iap.StochasticParameter:
            maxval_norm = maxval / m_max
            return iap.Multiply(level, maxval_norm, elementwise=True)

        def _int_parameter(
            level: iap.StochasticParameter, maxval: float
        ) -> iap.StochasticParameter:
            # paper applies just int(), so we don't round here
            return iap.Discretize(_float_parameter(level, maxval), round=False)

        # In the paper's code they use the definition from AutoAugment,
        # which is 0.1 + M*1.8/10. But that results in 0.1 for M=0, i.e. for
        # Brightness an almost black image, while M=5 would result in an
        # unaltered image. For AutoAugment that may be fine, as M is optimized
        # for each operation individually, but here we have only one fixed M
        # for all operations. Hence, we rather set this to 1.0 +/- M*0.9/10,
        # so that M=10 would result in 0.1 or 1.9.
        def _enhance_parameter(level: iap.StochasticParameter) -> iap.StochasticParameter:
            fparam = _float_parameter(level, 0.9)
            return iap.Clip(iap.Add(1.0, iap.RandomSign(fparam), elementwise=True), 0.1, 1.9)

        def _subtract(a: Numberish, b: Numberish) -> iap.StochasticParameter:
            return iap.Subtract(a, b, elementwise=True)

        def _affine(*args: object, **kwargs: object) -> pillike.Affine:
            kwargs = dict(kwargs)
            kwargs["fillcolor"] = cval
            if "center" not in kwargs:
                kwargs["center"] = (0.0, 0.0)
            return pillike.Affine(*args, **kwargs)

        _rnd_s = iap.RandomSign
        shear_max = np.rad2deg(0.3)

        # we don't add vertical flips here, paper is not really clear about
        # whether they used them or not
        return [
            meta.Identity(),
            pillike.Autocontrast(cutoff=0),
            pillike.Equalize(),
            arithmetic.Invert(p=1.0),
            # they use Image.rotate() for the rotation, which uses
            # the image center as the rotation center
            _affine(rotate=_rnd_s(_float_parameter(m, 30)), center=(0.5, 0.5)),
            # paper uses 4 - int_parameter(M, 4)
            pillike.Posterize(nb_bits=_subtract(8, iap.Clip(_int_parameter(m, 6), 0, 6))),
            # paper uses 256 - int_parameter(M, 256)
            pillike.Solarize(
                p=1.0, threshold=iap.Clip(_subtract(256, _int_parameter(m, 256)), 0, 256)
            ),
            pillike.EnhanceColor(_enhance_parameter(m)),
            pillike.EnhanceContrast(_enhance_parameter(m)),
            pillike.EnhanceBrightness(_enhance_parameter(m)),
            pillike.EnhanceSharpness(_enhance_parameter(m)),
            _affine(shear={"x": _rnd_s(_float_parameter(m, shear_max))}),
            _affine(shear={"y": _rnd_s(_float_parameter(m, shear_max))}),
            _affine(translate_percent={"x": _rnd_s(_float_parameter(m, 0.33))}),
            _affine(translate_percent={"y": _rnd_s(_float_parameter(m, 0.33))}),
            # paper code uses 20px on CIFAR (i.e. size 20/32), no information
            # on ImageNet values so we just use the same values
            arithmetic.Cutout(
                1,
                size=iap.Clip(_float_parameter(m, 20 / 32), 0, 20 / 32),
                squared=True,
                fill_mode="constant",
                cval=cval,
            ),
            pillike.FilterBlur(),
            pillike.FilterSmooth(),
        ]

    @legacy(version="0.4.0")
    def get_parameters(self) -> list[iap.StochasticParameter | int | Sequence[int]]:
        """See :func:`~imgaug2.augmenters.meta.Augmenter.get_parameters`."""
        someof = self[1]
        return [someof.n, self._m, self._cval]


class PosePreset(meta.Sequential):
    """
    Pose estimation augmentation preset built from common toolchain configs.

    This preset is based on augmentation settings documented for Lightning Pose,
    DeepLabCut (PyTorch/TF), MMPose, and SLEAP. Reference snippets are stored in
    ``third_party/pose_presets``.
    """

    def __init__(
        self,
        preset: PosePresetName = "lightning_pose_dlc",
        random_order: bool = False,
        seed: RNGInput = None,
        name: str | None = None,
        random_state: RNGInput | Literal["deprecated"] = "deprecated",
        deterministic: bool | Literal["deprecated"] = "deprecated",
    ) -> None:
        children = _pose_preset_children(preset)
        super().__init__(
            children=children,
            random_order=random_order,
            seed=seed,
            name=name,
            random_state=random_state,
            deterministic=deterministic,
        )
        self.preset = preset


def _pose_preset_children(preset: PosePresetName) -> list[meta.Augmenter]:
    if preset in {"lightning_pose_dlc", "deeplabcut_pytorch_default"}:
        return _pose_preset_children_dlc(hflip_prob=0.0, vflip_prob=0.0)
    if preset == "lightning_pose_dlc_lr":
        return _pose_preset_children_dlc(hflip_prob=0.5, vflip_prob=0.0)
    if preset == "lightning_pose_dlc_top_down":
        return _pose_preset_children_dlc(hflip_prob=0.5, vflip_prob=0.5)
    if preset == "sleap_default":
        return _pose_preset_children_sleap()
    if preset == "mmpose_default":
        return _pose_preset_children_mmpose()
    raise ValueError(f"Unknown PosePreset preset: {preset}")


def _pose_preset_children_dlc(*, hflip_prob: float, vflip_prob: float) -> list[meta.Augmenter]:
    rotation_range = 20.0
    zoom_range = 0.15
    shift_range = 0.1
    rotation_prob = 0.4
    zoom_prob = 0.4
    shift_prob = 0.3
    covering_prob = 0.5
    covering_ratio = 0.1
    covering_radius = 5
    gaussian_noise = 0.01

    rot = iap.Binomial(rotation_prob) * iap.Uniform(-rotation_range, rotation_range)
    scale = 1.0 + iap.Binomial(zoom_prob) * iap.Uniform(-zoom_range, zoom_range)
    shift = iap.Binomial(shift_prob) * iap.Uniform(-shift_range, shift_range)

    augs: list[meta.Augmenter] = []
    if hflip_prob > 0.0:
        augs.append(flip.Fliplr(hflip_prob))
    if vflip_prob > 0.0:
        augs.append(flip.Flipud(vflip_prob))

    augs.append(
        geometric.Affine(
            rotate=rot,
            scale=scale,
            translate_percent={"x": shift, "y": shift},
            mode="reflect",
            order=1,
        )
    )
    augs.append(
        meta.Sometimes(
            covering_prob,
            arithmetic.CoarseDropout(p=covering_ratio, size_px=covering_radius),
        )
    )
    if gaussian_noise > 0.0:
        augs.append(arithmetic.AdditiveGaussianNoise(scale=gaussian_noise * 255.0))
    return augs


def _pose_preset_children_sleap() -> list[meta.Augmenter]:
    apply_prob = 0.5
    rotate = 180.0
    translate = 50
    scale = 0.2
    uniform_noise = 0.02
    gaussian_noise = 5.0
    contrast_delta = 0.1
    brightness_delta = 0.2

    seq = meta.Sequential(
        [
            geometric.Affine(
                rotate=(-rotate, rotate),
                scale=(1.0 - scale, 1.0 + scale),
                translate_px={"x": (-translate, translate), "y": (-translate, translate)},
                mode="reflect",
                order=1,
            ),
            arithmetic.Add((-uniform_noise * 255.0, uniform_noise * 255.0)),
            arithmetic.AdditiveGaussianNoise(scale=gaussian_noise),
            contrast.LinearContrast((1.0 - contrast_delta, 1.0 + contrast_delta)),
            arithmetic.Multiply((1.0 - brightness_delta, 1.0 + brightness_delta)),
        ],
        random_order=False,
    )
    return [meta.Sometimes(apply_prob, seq)]


def _pose_preset_children_mmpose() -> list[meta.Augmenter]:
    return [
        flip.Fliplr(0.5),
        geometric.Affine(
            rotate=(-80, 80),
            scale=(1.0 - 0.25, 1.0 + 0.25),
            translate_percent={"x": (-0.16, 0.16), "y": (-0.16, 0.16)},
            mode="reflect",
            order=1,
        ),
    ]
