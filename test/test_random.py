import copy as copylib
import unittest
from unittest import mock

import numpy as np

import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.random as iarandom
from imgaug2.testutils import reseed



class _Base(unittest.TestCase):
    def setUp(self):
        reseed()


class TestConstants(_Base):
    def test_global_rng(self):
        iarandom.get_global_rng()  # creates global RNG upon first call
        assert iarandom.GLOBAL_RNG is not None


class TestRNG(_Base):
    @mock.patch("imgaug2.random.normalize_generator_")
    def test___init___calls_normalize_mocked(self, mock_norm):
        _ = iarandom.RNG(0)
        mock_norm.assert_called_once_with(0)

    def test___init___with_rng(self):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(rng1)

        assert rng2.generator is rng1.generator

    @mock.patch("imgaug2.random.get_generator_state")
    def test_state_getter_mocked(self, mock_get):
        mock_get.return_value = "mock"
        rng = iarandom.RNG(0)
        result = rng.state
        assert result == "mock"
        mock_get.assert_called_once_with(rng.generator)

    @mock.patch("imgaug2.random.RNG.set_state_")
    def test_state_setter_mocked(self, mock_set):
        rng = iarandom.RNG(0)
        state = {"state": 123}
        rng.state = state
        mock_set.assert_called_once_with(state)

    @mock.patch("imgaug2.random.set_generator_state_")
    def test_set_state__mocked(self, mock_set):
        rng = iarandom.RNG(0)
        state = {"state": 123}
        result = rng.set_state_(state)
        assert result is rng
        mock_set.assert_called_once_with(rng.generator, state)

    @mock.patch("imgaug2.random.set_generator_state_")
    def test_use_state_of__mocked(self, mock_set):
        rng1 = iarandom.RNG(0)
        rng2 = mock.MagicMock()
        state = {"state": 123}
        rng2.state = state
        result = rng1.use_state_of_(rng2)
        assert result == rng1
        mock_set.assert_called_once_with(rng1.generator, state)

    @mock.patch("imgaug2.random.get_global_rng")
    def test_is_global__is_global__rng_mocked(self, mock_get):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(rng1.generator)
        mock_get.return_value = rng2
        assert rng1.is_global_rng() is True

    @mock.patch("imgaug2.random.get_global_rng")
    def test_is_global_rng__is_not_global__mocked(self, mock_get):
        rng1 = iarandom.RNG(0)
        # different instance with same state/seed should still be viewed as
        # different by the method
        rng2 = iarandom.RNG(0)
        mock_get.return_value = rng2
        assert rng1.is_global_rng() is False

    @mock.patch("imgaug2.random.get_global_rng")
    def test_equals_global_rng__is_global__mocked(self, mock_get):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(0)
        mock_get.return_value = rng2
        assert rng1.equals_global_rng() is True

    @mock.patch("imgaug2.random.get_global_rng")
    def test_equals_global_rng__is_not_global__mocked(self, mock_get):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(1)
        mock_get.return_value = rng2
        assert rng1.equals_global_rng() is False

    @mock.patch("imgaug2.random.generate_seed_")
    def test_generate_seed__mocked(self, mock_gen):
        rng = iarandom.RNG(0)
        mock_gen.return_value = -1
        seed = rng.generate_seed_()
        assert seed == -1
        mock_gen.assert_called_once_with(rng.generator)

    @mock.patch("imgaug2.random.generate_seeds_")
    def test_generate_seeds__mocked(self, mock_gen):
        rng = iarandom.RNG(0)
        mock_gen.return_value = [-1, -2]
        seeds = rng.generate_seeds_(2)
        assert seeds == [-1, -2]
        mock_gen.assert_called_once_with(rng.generator, 2)

    @mock.patch("imgaug2.random.reset_generator_cache_")
    def test_reset_cache__mocked(self, mock_reset):
        rng = iarandom.RNG(0)
        result = rng.reset_cache_()
        assert result is rng
        mock_reset.assert_called_once_with(rng.generator)

    @mock.patch("imgaug2.random.derive_generators_")
    def test_derive_rng__mocked(self, mock_derive):
        gen = iarandom.convert_seed_to_generator(0)
        mock_derive.return_value = [gen]
        rng = iarandom.RNG(0)
        result = rng.derive_rng_()
        assert result.generator is gen
        mock_derive.assert_called_once_with(rng.generator, 1)

    @mock.patch("imgaug2.random.derive_generators_")
    def test_derive_rngs__mocked(self, mock_derive):
        gen1 = iarandom.convert_seed_to_generator(0)
        gen2 = iarandom.convert_seed_to_generator(1)
        mock_derive.return_value = [gen1, gen2]
        rng = iarandom.RNG(0)
        result = rng.derive_rngs_(2)
        assert result[0].generator is gen1
        assert result[1].generator is gen2
        mock_derive.assert_called_once_with(rng.generator, 2)

    @mock.patch("imgaug2.random.is_generator_equal_to")
    def test_equals_mocked(self, mock_equal):
        mock_equal.return_value = "foo"
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(1)
        result = rng1.equals(rng2)
        assert result == "foo"
        mock_equal.assert_called_once_with(rng1.generator, rng2.generator)

    def test_equals_identical_generators(self):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(rng1)
        assert rng1.equals(rng2)

    def test_equals_with_similar_generators(self):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(0)
        assert rng1.equals(rng2)

    def test_equals_with_different_generators(self):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(1)
        assert not rng1.equals(rng2)

    def test_equals_with_advanced_generator(self):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(0)
        rng2.advance_()
        assert not rng1.equals(rng2)

    @mock.patch("imgaug2.random.advance_generator_")
    def test_advance__mocked(self, mock_advance):
        rng = iarandom.RNG(0)
        result = rng.advance_()
        assert result is rng
        mock_advance.assert_called_once_with(rng.generator)

    @mock.patch("imgaug2.random.copy_generator")
    def test_copy_mocked(self, mock_copy):
        rng1 = iarandom.RNG(0)
        rng2 = iarandom.RNG(1)
        mock_copy.return_value = rng2.generator
        result = rng1.copy()
        assert result.generator is rng2.generator
        mock_copy.assert_called_once_with(rng1.generator)

    @mock.patch("imgaug2.random.RNG.copy")
    @mock.patch("imgaug2.random.RNG.is_global_rng")
    def test_copy_unless_global_rng__is_global__mocked(self, mock_is_global, mock_copy):
        rng = iarandom.RNG(0)
        mock_is_global.return_value = True
        mock_copy.return_value = "foo"
        result = rng.copy_unless_global_rng()
        assert result is rng
        mock_is_global.assert_called_once_with()
        assert mock_copy.call_count == 0

    @mock.patch("imgaug2.random.RNG.copy")
    @mock.patch("imgaug2.random.RNG.is_global_rng")
    def test_copy_unless_global_rng__is_not_global__mocked(self, mock_is_global, mock_copy):
        rng = iarandom.RNG(0)
        mock_is_global.return_value = False
        mock_copy.return_value = "foo"
        result = rng.copy_unless_global_rng()
        assert result is "foo"
        mock_is_global.assert_called_once_with()
        mock_copy.assert_called_once_with()

    def test_duplicate(self):
        rng = iarandom.RNG(0)
        rngs = rng.duplicate(1)
        assert rngs == [rng]

    def test_duplicate_two_entries(self):
        rng = iarandom.RNG(0)
        rngs = rng.duplicate(2)
        assert rngs == [rng, rng]

    @mock.patch("imgaug2.random.create_fully_random_generator")
    def test_create_fully_random_mocked(self, mock_create):
        gen = iarandom.convert_seed_to_generator(0)
        mock_create.return_value = gen
        rng = iarandom.RNG.create_fully_random()
        mock_create.assert_called_once_with()
        assert rng.generator is gen

    @mock.patch("imgaug2.random.derive_generators_")
    def test_create_pseudo_random__mocked(self, mock_get):
        rng_glob = iarandom.get_global_rng()
        rng = iarandom.RNG(0)
        mock_get.return_value = [rng.generator]
        result = iarandom.RNG.create_pseudo_random_()
        assert result.generator is rng.generator
        mock_get.assert_called_once_with(rng_glob.generator, 1)

    def test_integers_mocked(self):
        mock_gen = mock.MagicMock()
        mock_gen.integers.return_value = "foo"
        rng = iarandom.RNG(0)
        rng.generator = mock_gen

        result = rng.integers(low=0, high=1, size=(1,), dtype="int64", endpoint=True)

        assert result == "foo"
        mock_gen.integers.assert_called_once_with(
            low=0, high=1, size=(1,), dtype="int64", endpoint=True
        )

    def test_random_mocked(self):
        mock_gen = mock.MagicMock()
        mock_gen.random.return_value = "foo"
        rng = iarandom.RNG(0)
        rng.generator = mock_gen
        out = np.zeros((1,), dtype="float64")

        result = rng.random(size=(1,), dtype="float64", out=out)

        assert result == "foo"
        mock_gen.random.assert_called_once_with(size=(1,), dtype="float64", out=out)

    def test_generator_methods_non_mocked(self):
        float_specs = [
            ("beta", dict(a=1.0, b=2.0, size=(2,)), (2,)),
            ("chisquare", dict(df=2.0, size=(2,)), (2,)),
            ("dirichlet", dict(alpha=[0.2, 0.3, 0.5], size=1), (1, 3)),
            ("exponential", dict(scale=1.1, size=(2,)), (2,)),
            ("f", dict(dfnum=1.0, dfden=2.0, size=(2,)), (2,)),
            ("gamma", dict(shape=1.0, scale=1.2, size=(2,)), (2,)),
            ("gumbel", dict(loc=0.1, scale=1.1, size=(2,)), (2,)),
            ("laplace", dict(loc=0.5, scale=1.5, size=(2,)), (2,)),
            ("logistic", dict(loc=0.5, scale=1.5, size=(2,)), (2,)),
            ("lognormal", dict(mean=0.5, sigma=1.5, size=(2,)), (2,)),
            (
                "multivariate_normal",
                dict(mean=[0.0, 1.0], cov=[[1.0, 0.0], [0.0, 1.0]], size=2),
                (2, 2),
            ),
            ("noncentral_chisquare", dict(df=0.5, nonc=1.0, size=(2,)), (2,)),
            ("noncentral_f", dict(dfnum=0.5, dfden=1.5, nonc=2.0, size=(2,)), (2,)),
            ("normal", dict(loc=0.5, scale=1.0, size=(2,)), (2,)),
            ("pareto", dict(a=0.5, size=(2,)), (2,)),
            ("power", dict(a=0.5, size=(2,)), (2,)),
            ("rayleigh", dict(scale=1.5, size=(2,)), (2,)),
            ("standard_cauchy", dict(size=(2,)), (2,)),
            ("standard_exponential", dict(size=(2,)), (2,)),
            ("standard_gamma", dict(shape=1.0, size=(2,)), (2,)),
            ("standard_normal", dict(size=(2,)), (2,)),
            ("standard_t", dict(df=1.5, size=(2,)), (2,)),
            ("triangular", dict(left=1.0, mode=1.5, right=2.0, size=(2,)), (2,)),
            ("uniform", dict(low=0.5, high=1.5, size=(2,)), (2,)),
            ("vonmises", dict(mu=1.0, kappa=1.5, size=(2,)), (2,)),
            ("wald", dict(mean=0.5, scale=1.0, size=(2,)), (2,)),
            ("weibull", dict(a=1.0, size=(2,)), (2,)),
        ]
        int_specs = [
            ("binomial", dict(n=10, p=0.1, size=(2,)), (2,)),
            ("geometric", dict(p=0.5, size=(2,)), (2,)),
            ("hypergeometric", dict(ngood=2, nbad=4, nsample=3, size=(2,)), (2,)),
            ("logseries", dict(p=0.5, size=(2,)), (2,)),
            ("multinomial", dict(n=5, pvals=[0.2, 0.3, 0.5], size=2), (2, 3)),
            ("negative_binomial", dict(n=10, p=0.5, size=(2,)), (2,)),
            ("poisson", dict(lam=1.5, size=(2,)), (2,)),
            ("zipf", dict(a=1.0, size=(2,)), (2,)),
        ]

        for name, kwargs, expected_shape in float_specs:
            with self.subTest(name=name):
                rng = iarandom.RNG(0)
                result = getattr(rng, name)(**kwargs)
                arr = np.asarray(result)
                assert arr.shape == expected_shape
                assert arr.dtype.kind == "f"
                assert np.all(np.isfinite(arr))

        for name, kwargs, expected_shape in int_specs:
            with self.subTest(name=name):
                rng = iarandom.RNG(0)
                result = getattr(rng, name)(**kwargs)
                arr = np.asarray(result)
                assert arr.shape == expected_shape
                assert arr.dtype.kind in ["i", "u"]
                assert np.all(np.isfinite(arr))

        rng = iarandom.RNG(0)
        result = rng.standard_exponential(size=(2,))
        assert result.dtype.name == "float32"
        result = rng.standard_gamma(shape=1.0, size=(2,))
        assert result.dtype.name == "float32"
        result = rng.standard_normal(size=(2,))
        assert result.dtype.name == "float32"

    def test_choice_bytes_shuffle_permutation_non_mocked(self):
        rng = iarandom.RNG(0)

        choice = rng.choice([1, 2, 3, 4], size=(2,), replace=False)
        assert choice.shape == (2,)
        assert set(choice.tolist()).issubset({1, 2, 3, 4})

        result_bytes = rng.bytes(length=10)
        assert isinstance(result_bytes, (bytes, bytearray))
        assert len(result_bytes) == 10

        arr = [0, 1, 2, 3, 4]
        rng.shuffle(arr)
        assert sorted(arr) == [0, 1, 2, 3, 4]

        perm = rng.permutation([1, 2, 3, 4])
        assert perm.shape == (4,)
        assert set(perm.tolist()) == {1, 2, 3, 4}

    def test_choice_mocked(self):
        self._test_sampling_func("choice", a=[1, 2, 3], size=(1,), replace=False, p=[0.1, 0.2, 0.7])

    def test_bytes_mocked(self):
        self._test_sampling_func("bytes", length=[10])

    def test_shuffle_mocked(self):
        mock_gen = mock.MagicMock()
        rng = iarandom.RNG(0)
        rng.generator = mock_gen

        rng.shuffle([1, 2, 3])

        mock_gen.shuffle.assert_called_once_with([1, 2, 3])

    def test_permutation_mocked(self):
        mock_gen = mock.MagicMock()
        rng = iarandom.RNG(0)
        rng.generator = mock_gen
        mock_gen.permutation.return_value = "foo"

        result = rng.permutation([1, 2, 3])

        assert result == "foo"
        mock_gen.permutation.assert_called_once_with([1, 2, 3])

    def test_beta_mocked(self):
        self._test_sampling_func("beta", a=1.0, b=2.0, size=(1,))

    def test_binomial_mocked(self):
        self._test_sampling_func("binomial", n=10, p=0.1, size=(1,))

    def test_chisquare_mocked(self):
        self._test_sampling_func("chisquare", df=2, size=(1,))

    def test_dirichlet_mocked(self):
        self._test_sampling_func("dirichlet", alpha=0.1, size=(1,))

    def test_exponential_mocked(self):
        self._test_sampling_func("exponential", scale=1.1, size=(1,))

    def test_f_mocked(self):
        self._test_sampling_func("f", dfnum=1, dfden=2, size=(1,))

    def test_gamma_mocked(self):
        self._test_sampling_func("gamma", shape=1, scale=1.2, size=(1,))

    def test_geometric_mocked(self):
        self._test_sampling_func("geometric", p=0.5, size=(1,))

    def test_gumbel_mocked(self):
        self._test_sampling_func("gumbel", loc=0.1, scale=1.1, size=(1,))

    def test_hypergeometric_mocked(self):
        self._test_sampling_func("hypergeometric", ngood=2, nbad=4, nsample=6, size=(1,))

    def test_laplace_mocked(self):
        self._test_sampling_func("laplace", loc=0.5, scale=1.5, size=(1,))

    def test_logistic_mocked(self):
        self._test_sampling_func("logistic", loc=0.5, scale=1.5, size=(1,))

    def test_lognormal_mocked(self):
        self._test_sampling_func("lognormal", mean=0.5, sigma=1.5, size=(1,))

    def test_logseries_mocked(self):
        self._test_sampling_func("logseries", p=0.5, size=(1,))

    def test_multinomial_mocked(self):
        self._test_sampling_func("multinomial", n=5, pvals=0.5, size=(1,))

    def test_multivariate_normal_mocked(self):
        self._test_sampling_func(
            "multivariate_normal", mean=0.5, cov=1.0, size=(1,), check_valid="foo", tol=1e-2
        )

    def test_negative_binomial_mocked(self):
        self._test_sampling_func("negative_binomial", n=10, p=0.5, size=(1,))

    def test_noncentral_chisquare_mocked(self):
        self._test_sampling_func("noncentral_chisquare", df=0.5, nonc=1.0, size=(1,))

    def test_noncentral_f_mocked(self):
        self._test_sampling_func("noncentral_f", dfnum=0.5, dfden=1.5, nonc=2.0, size=(1,))

    def test_normal_mocked(self):
        self._test_sampling_func("normal", loc=0.5, scale=1.0, size=(1,))

    def test_pareto_mocked(self):
        self._test_sampling_func("pareto", a=0.5, size=(1,))

    def test_poisson_mocked(self):
        self._test_sampling_func("poisson", lam=1.5, size=(1,))

    def test_power_mocked(self):
        self._test_sampling_func("power", a=0.5, size=(1,))

    def test_rayleigh_mocked(self):
        self._test_sampling_func("rayleigh", scale=1.5, size=(1,))

    def test_standard_cauchy_mocked(self):
        self._test_sampling_func("standard_cauchy", size=(1,))

    def test_standard_exponential_np117_mocked(self):
        fname = "standard_exponential"

        arr = np.zeros((1,), dtype="float16")
        args = []
        kwargs = {"size": (1,), "dtype": "float16", "method": "foo", "out": arr}

        mock_gen = mock.MagicMock()
        getattr(mock_gen, fname).return_value = "foo"
        rng = iarandom.RNG(0)
        rng.generator = mock_gen

        result = getattr(rng, fname)(*args, **kwargs)

        assert result == "foo"
        getattr(mock_gen, fname).assert_called_once_with(*args, **kwargs)

    def test_standard_gamma_np117_mocked(self):
        fname = "standard_gamma"

        arr = np.zeros((1,), dtype="float16")
        args = []
        kwargs = {"shape": 1.0, "size": (1,), "dtype": "float16", "out": arr}

        mock_gen = mock.MagicMock()
        getattr(mock_gen, fname).return_value = "foo"
        rng = iarandom.RNG(0)
        rng.generator = mock_gen

        result = getattr(rng, fname)(*args, **kwargs)

        assert result == "foo"
        getattr(mock_gen, fname).assert_called_once_with(*args, **kwargs)

    def test_standard_normal_np117_mocked(self):
        fname = "standard_normal"

        arr = np.zeros((1,), dtype="float16")
        args = []
        kwargs = {"size": (1,), "dtype": "float16", "out": arr}

        mock_gen = mock.MagicMock()
        getattr(mock_gen, fname).return_value = "foo"
        rng = iarandom.RNG(0)
        rng.generator = mock_gen

        result = getattr(rng, fname)(*args, **kwargs)

        assert result == "foo"
        getattr(mock_gen, fname).assert_called_once_with(*args, **kwargs)

    def test_standard_t_mocked(self):
        self._test_sampling_func("standard_t", df=1.5, size=(1,))

    def test_triangular_mocked(self):
        self._test_sampling_func("triangular", left=1.0, mode=1.5, right=2.0, size=(1,))

    def test_uniform_mocked(self):
        self._test_sampling_func("uniform", low=0.5, high=1.5, size=(1,))

    def test_vonmises_mocked(self):
        self._test_sampling_func("vonmises", mu=1.0, kappa=1.5, size=(1,))

    def test_wald_mocked(self):
        self._test_sampling_func("wald", mean=0.5, scale=1.0, size=(1,))

    def test_weibull_mocked(self):
        self._test_sampling_func("weibull", a=1.0, size=(1,))

    def test_zipf_mocked(self):
        self._test_sampling_func("zipf", a=1.0, size=(1,))

    @classmethod
    def _test_sampling_func(cls, fname, *args, **kwargs):
        mock_gen = mock.MagicMock()
        getattr(mock_gen, fname).return_value = "foo"
        rng = iarandom.RNG(0)
        rng.generator = mock_gen

        result = getattr(rng, fname)(*args, **kwargs)

        assert result == "foo"
        getattr(mock_gen, fname).assert_called_once_with(*args, **kwargs)


class Test_get_global_rng(_Base):
    def test_call(self):
        iarandom.seed(0)

        rng = iarandom.get_global_rng()

        expected = iarandom.RNG(0)
        assert rng is not None
        assert rng.equals(expected)


class Test_seed(_Base):
    def test_integrationtest(self):
        iarandom.seed(1)
        assert iarandom.GLOBAL_RNG.equals(iarandom.RNG(1))

    def test_seed_affects_augmenters_created_after_its_call(self):
        image = np.full((50, 50, 3), 128, dtype=np.uint8)

        images_aug = []
        for _ in np.arange(5):
            iarandom.seed(100)
            aug = iaa.AdditiveGaussianNoise(scale=50, per_channel=True)
            images_aug.append(aug(image=image))

        # assert all images identical
        for other_image_aug in images_aug[1:]:
            assert np.array_equal(images_aug[0], other_image_aug)

        # but different seed must lead to different image
        iarandom.seed(101)
        aug = iaa.AdditiveGaussianNoise(scale=50, per_channel=True)
        image_aug = aug(image=image)
        assert not np.array_equal(images_aug[0], image_aug)

    def test_seed_affects_augmenters_created_before_its_call(self):
        image = np.full((50, 50, 3), 128, dtype=np.uint8)

        images_aug = []
        for _ in np.arange(5):
            aug = iaa.AdditiveGaussianNoise(scale=50, per_channel=True)
            iarandom.seed(100)
            images_aug.append(aug(image=image))

        # assert all images identical
        for other_image_aug in images_aug[1:]:
            assert np.array_equal(images_aug[0], other_image_aug)

        # but different seed must lead to different image
        aug = iaa.AdditiveGaussianNoise(scale=50, per_channel=True)
        iarandom.seed(101)
        image_aug = aug(image=image)
        assert not np.array_equal(images_aug[0], image_aug)


class Test_normalize_generator(_Base):
    @mock.patch("imgaug2.random.normalize_generator_")
    def test_mocked_call(self, mock_subfunc):
        mock_subfunc.return_value = "foo"
        inputs = ["bar"]

        result = iarandom.normalize_generator(inputs)

        assert mock_subfunc.call_count == 1
        assert mock_subfunc.call_args[0][0] is not inputs
        assert mock_subfunc.call_args[0][0] == inputs
        assert result == "foo"


class Test_normalize_generator_(_Base):
    def test_called_with_none(self):
        result = iarandom.normalize_generator_(None)
        assert result is iarandom.get_global_rng().generator

    def test_called_with_seed_sequence(self):
        seedseq = np.random.SeedSequence(0)

        result = iarandom.normalize_generator_(seedseq)

        expected = np.random.Generator(iarandom.BIT_GENERATOR(np.random.SeedSequence(0)))
        assert iarandom.is_generator_equal_to(result, expected)

    def test_called_with_bit_generator(self):
        bgen = iarandom.BIT_GENERATOR(np.random.SeedSequence(0))

        result = iarandom.normalize_generator_(bgen)

        assert result.bit_generator is bgen

    def test_called_with_generator(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(np.random.SeedSequence(0)))

        result = iarandom.normalize_generator_(gen)

        assert result is gen

    def test_called_int(self):
        seed = 0

        result = iarandom.normalize_generator_(seed)

        expected = iarandom.convert_seed_to_generator(seed)
        assert iarandom.is_generator_equal_to(result, expected)


class Test_convert_seed_to_generator(_Base):
    def test_call(self):
        gen = iarandom.convert_seed_to_generator(1)
        expected = np.random.Generator(iarandom.BIT_GENERATOR(np.random.SeedSequence(1)))
        assert iarandom.is_generator_equal_to(gen, expected)


class Test_convert_seed_sequence_to_generator(_Base):
    def test_call(self):
        seedseq = np.random.SeedSequence(1)

        gen = iarandom.convert_seed_sequence_to_generator(seedseq)

        expected = np.random.Generator(iarandom.BIT_GENERATOR(np.random.SeedSequence(1)))
        assert iarandom.is_generator_equal_to(gen, expected)


class Test_create_pseudo_random_generator_(_Base):
    def test_call(self):
        global_gen = copylib.deepcopy(iarandom.get_global_rng().generator)

        gen = iarandom.create_pseudo_random_generator_()

        expected = iarandom.convert_seed_to_generator(iarandom.generate_seed_(global_gen))
        assert iarandom.is_generator_equal_to(gen, expected)


class Test_create_fully_random_generator(_Base):
    def test_call(self):
        gen = iarandom.create_fully_random_generator()

        assert isinstance(gen, np.random.Generator)
        assert isinstance(gen.bit_generator, np.random.SFC64)


class Test_generate_seed_(_Base):
    @mock.patch("imgaug2.random.generate_seeds_")
    def test_mocked_call(self, mock_seeds):
        gen = iarandom.convert_seed_to_generator(0)

        _ = iarandom.generate_seed_(gen)

        mock_seeds.assert_called_once_with(gen, 1)


class Test_generate_seeds_(_Base):
    def test_call(self):
        gen = iarandom.convert_seed_to_generator(0)

        seeds = iarandom.generate_seeds_(gen, 2)

        assert len(seeds) == 2
        assert ia.is_np_array(seeds)
        assert seeds.dtype.name == "int32"


class Test_copy_generator(_Base):
    @mock.patch("imgaug2.random._copy_generator_np117")
    def test_mocked_call_with_generator(self, mock_np117):
        mock_np117.return_value = "np117"
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        gen_copy = iarandom.copy_generator(gen)

        assert gen_copy == "np117"
        mock_np117.assert_called_once_with(gen)

    def test_call_with_generator(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        gen_copy = iarandom._copy_generator_np117(gen)

        assert gen is not gen_copy
        assert iarandom.is_generator_equal_to(gen, gen_copy)


class Test_copy_generator_unless_global_generator(_Base):
    @mock.patch("imgaug2.random.get_global_rng")
    @mock.patch("imgaug2.random.copy_generator")
    def test_mocked_gen_is_global(self, mock_copy, mock_get_global_rng):
        gen = iarandom.convert_seed_to_generator(1)
        mock_copy.return_value = "foo"
        mock_get_global_rng.return_value = iarandom.RNG(gen)

        result = iarandom.copy_generator_unless_global_generator(gen)

        assert mock_get_global_rng.call_count == 1
        assert mock_copy.call_count == 0
        assert result is gen

    @mock.patch("imgaug2.random.get_global_rng")
    @mock.patch("imgaug2.random.copy_generator")
    def test_mocked_gen_is_not_global(self, mock_copy, mock_get_global_rng):
        gen1 = iarandom.convert_seed_to_generator(1)
        gen2 = iarandom.convert_seed_to_generator(2)
        mock_copy.return_value = "foo"
        mock_get_global_rng.return_value = iarandom.RNG(gen2)

        result = iarandom.copy_generator_unless_global_generator(gen1)

        assert mock_get_global_rng.call_count == 1
        mock_copy.assert_called_once_with(gen1)
        assert result == "foo"


class Test_reset_generator_cache_(_Base):
    @mock.patch("imgaug2.random._reset_generator_cache_np117_")
    def test_mocked_call(self, mock_np117):
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.reset_generator_cache_(gen)

        assert result == "np117"
        mock_np117.assert_called_once_with(gen)

    def test_call(self):
        gen = iarandom.convert_seed_to_generator(1)
        gen_without_cache_copy = copylib.deepcopy(gen)

        state = iarandom._get_generator_state_np117(gen)
        state["has_uint32"] = 1
        gen_with_cache = copylib.deepcopy(gen)
        iarandom.set_generator_state_(gen_with_cache, state)
        gen_with_cache_copy = copylib.deepcopy(gen_with_cache)

        gen_cache_reset = iarandom.reset_generator_cache_(gen_with_cache)

        assert iarandom.is_generator_equal_to(gen_cache_reset, gen_without_cache_copy)
        assert not iarandom.is_generator_equal_to(gen_cache_reset, gen_with_cache_copy)


class Test_derive_generator_(_Base):
    @mock.patch("imgaug2.random.derive_generators_")
    def test_mocked_call(self, mock_derive_gens):
        mock_derive_gens.return_value = ["foo"]
        gen = iarandom.convert_seed_to_generator(1)

        gen_derived = iarandom.derive_generator_(gen)

        mock_derive_gens.assert_called_once_with(gen, n=1)
        assert gen_derived == "foo"

    def test_integration(self):
        gen = iarandom.convert_seed_to_generator(1)
        gen_copy = copylib.deepcopy(gen)

        gen_derived = iarandom.derive_generator_(gen)

        assert not iarandom.is_generator_equal_to(gen_derived, gen_copy)
        # should have advanced the state
        assert not iarandom.is_generator_equal_to(gen_copy, gen)


class Test_derive_generators_(_Base):
    @mock.patch("imgaug2.random._derive_generators_np117_")
    def test_mocked_call(self, mock_np117):
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.derive_generators_(gen, 1)

        assert result == "np117"
        mock_np117.assert_called_once_with(gen, n=1)

    def test_call(self):
        gen = iarandom.convert_seed_to_generator(1)
        gen_copy = copylib.deepcopy(gen)

        result = iarandom.derive_generators_(gen, 2)

        assert len(result) == 2
        assert np.all([isinstance(gen, np.random.Generator) for gen in result])
        assert not iarandom.is_generator_equal_to(result[0], gen_copy)
        assert not iarandom.is_generator_equal_to(result[1], gen_copy)
        assert not iarandom.is_generator_equal_to(result[0], result[1])
        # derive should advance state
        assert not iarandom.is_generator_equal_to(gen, gen_copy)


class Test_get_generator_state(_Base):
    @mock.patch("imgaug2.random._get_generator_state_np117")
    def test_mocked_call(self, mock_np117):
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.get_generator_state(gen)

        assert result == "np117"
        mock_np117.assert_called_once_with(gen)

    def test_call(self):
        gen = iarandom.convert_seed_to_generator(1)
        state = iarandom.get_generator_state(gen)
        assert str(state) == str(gen.bit_generator.state)


class Test_set_generator_state_(_Base):
    @mock.patch("imgaug2.random._set_generator_state_np117_")
    def test_mocked_call(self, mock_np117):
        gen = iarandom.convert_seed_to_generator(1)
        state = {"state": 0}

        iarandom.set_generator_state_(gen, state)

        mock_np117.assert_called_once_with(gen, state)

    def test_call(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))
        gen1_copy = copylib.deepcopy(gen1)
        gen2_copy = copylib.deepcopy(gen2)

        iarandom._set_generator_state_np117_(gen2, iarandom.get_generator_state(gen1))

        assert iarandom.is_generator_equal_to(gen2, gen1)
        assert iarandom.is_generator_equal_to(gen1, gen1_copy)
        assert not iarandom.is_generator_equal_to(gen2, gen2_copy)

    def test_call_via_samples(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))
        gen1_copy = copylib.deepcopy(gen1)
        gen2_copy = copylib.deepcopy(gen2)

        iarandom._set_generator_state_np117_(gen2, iarandom.get_generator_state(gen1))

        samples1 = gen1.random(size=(100,))
        samples2 = gen2.random(size=(100,))
        samples1_copy = gen1_copy.random(size=(100,))
        samples2_copy = gen2_copy.random(size=(100,))

        assert np.allclose(samples1, samples2)
        assert np.allclose(samples1, samples1_copy)
        assert not np.allclose(samples2, samples2_copy)


class Test_is_generator_equal_to(_Base):
    @mock.patch("imgaug2.random._is_generator_equal_to_np117")
    def test_mocked_call(self, mock_np117):
        mock_np117.return_value = "np117"
        gen = iarandom.convert_seed_to_generator(1)

        result = iarandom.is_generator_equal_to(gen, gen)

        assert result == "np117"
        mock_np117.assert_called_once_with(gen, gen)

    def test_generator_is_identical(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom._is_generator_equal_to_np117(gen, gen)

        assert result is True

    def test_generator_created_with_same_seed(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom._is_generator_equal_to_np117(gen1, gen2)

        assert result is True

    def test_generator_is_copy_of_itself(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))

        result = iarandom._is_generator_equal_to_np117(gen1, copylib.deepcopy(gen1))

        assert result is True

    def test_generator_created_with_different_seed(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))

        result = iarandom._is_generator_equal_to_np117(gen1, gen2)

        assert result is False

    def test_generator_modified_to_have_same_state(self):
        gen1 = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen2 = np.random.Generator(iarandom.BIT_GENERATOR(2))
        iarandom.set_generator_state_(gen2, iarandom.get_generator_state(gen1))

        result = iarandom._is_generator_equal_to_np117(gen1, gen2)

        assert result is True


class Test_advance_generator_(_Base):
    @mock.patch("imgaug2.random._advance_generator_np117_")
    def test_mocked_call(self, mock_np117):
        gen = iarandom.convert_seed_to_generator(1)

        iarandom.advance_generator_(gen)

        mock_np117.assert_called_once_with(gen)

    def test_call(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen_copy1 = copylib.deepcopy(gen)

        iarandom._advance_generator_np117_(gen)
        gen_copy2 = copylib.deepcopy(gen)

        iarandom._advance_generator_np117_(gen)

        assert iarandom.is_generator_equal_to(gen, copylib.deepcopy(gen))
        assert not iarandom.is_generator_equal_to(gen_copy1, gen_copy2)
        assert not iarandom.is_generator_equal_to(gen_copy2, gen)
        assert not iarandom.is_generator_equal_to(gen_copy1, gen)
    def test_samples_different_after_advance(self):
        gen = np.random.Generator(iarandom.BIT_GENERATOR(1))
        gen_copy1 = copylib.deepcopy(gen)

        # first advance
        iarandom._advance_generator_np117_(gen)
        gen_copy2 = copylib.deepcopy(gen)

        # second advance
        iarandom._advance_generator_np117_(gen)

        sample_before = gen_copy1.uniform(0.0, 1.0)
        sample_after = gen_copy2.uniform(0.0, 1.0)
        sample_after_after = gen.uniform(0.0, 1.0)
        assert not np.isclose(sample_after, sample_before, rtol=0)
        assert not np.isclose(sample_after_after, sample_after, rtol=0)
        assert not np.isclose(sample_after_after, sample_before, rtol=0)


class Test_temporary_numpy_seed(_Base):
    def test_seed_is_applied_inside_context(self):
        # Get values with seed 42
        np.random.seed(42)
        expected = np.random.rand(5)

        # Use context to get same values
        with iarandom.temporary_numpy_seed(42):
            result = np.random.rand(5)

        assert np.allclose(result, expected)

    def test_state_is_restored_after_context(self):
        # Set a known state
        np.random.seed(123)
        before = np.random.rand(3)

        # Reset to same state
        np.random.seed(123)
        np.random.rand(3)  # advance past first 3

        # Use context with different seed
        with iarandom.temporary_numpy_seed(999):
            _ = np.random.rand(10)  # advance inside context

        # After context, state should be restored - get next 3 values
        after = np.random.rand(3)

        # Reset and verify
        np.random.seed(123)
        _ = np.random.rand(3)  # skip first 3
        expected_after = np.random.rand(3)

        assert np.allclose(after, expected_after)

    def test_none_entropy_does_nothing(self):
        # Set a known state
        np.random.seed(42)
        expected = np.random.rand(5)

        # Reset
        np.random.seed(42)

        # Context with None should not alter state
        with iarandom.temporary_numpy_seed(None):
            result = np.random.rand(5)

        assert np.allclose(result, expected)

    def test_reproducible_results_with_same_seed(self):
        results = []
        for _ in range(3):
            with iarandom.temporary_numpy_seed(12345):
                results.append(np.random.rand(10))

        assert np.allclose(results[0], results[1])
        assert np.allclose(results[1], results[2])
