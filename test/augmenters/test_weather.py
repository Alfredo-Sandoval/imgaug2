import unittest
from unittest import mock

import cv2
import numpy as np

import imgaug2 as ia
from imgaug2 import augmenters as iaa
from imgaug2 import parameters as iap
from imgaug2.testutils import is_parameter_instance, reseed, runtest_pickleable_uint8_img


class _TwoValueParam(iap.StochasticParameter):
    def __init__(self, v1, v2):
        super().__init__()
        self.v1 = v1
        self.v2 = v2

    def _draw_samples(self, size, random_state):
        arr = np.full(size, self.v1, dtype=np.float32)
        arr[1::2] = self.v2
        return arr


class TestFastSnowyLandscape(unittest.TestCase):
    def setUp(self):
        reseed()

    def test___init__(self):
        # check parameters
        aug = iaa.FastSnowyLandscape(
            lightness_threshold=[100, 200], lightness_multiplier=[1.0, 4.0]
        )
        assert is_parameter_instance(aug.lightness_threshold, iap.Choice)
        assert len(aug.lightness_threshold.a) == 2
        assert aug.lightness_threshold.a[0] == 100
        assert aug.lightness_threshold.a[1] == 200

        assert is_parameter_instance(aug.lightness_multiplier, iap.Choice)
        assert len(aug.lightness_multiplier.a) == 2
        assert np.allclose(aug.lightness_multiplier.a[0], 1.0)
        assert np.allclose(aug.lightness_multiplier.a[1], 4.0)

    def test_basic_functionality(self):
        # basic functionality test
        aug = iaa.FastSnowyLandscape(lightness_threshold=100, lightness_multiplier=2.0)
        image = np.arange(0, 6 * 6 * 3).reshape((6, 6, 3)).astype(np.uint8)
        image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        mask = image_hls[..., 1] < 100
        expected = np.copy(image_hls).astype(np.float32)
        expected[..., 1][mask] *= 2.0
        expected = np.clip(np.round(expected), 0, 255).astype(np.uint8)
        expected = cv2.cvtColor(expected, cv2.COLOR_HLS2RGB)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, expected)

    def test_vary_lightness_threshold(self):
        # test when varying lightness_threshold between images
        image = np.arange(0, 6 * 6 * 3).reshape((6, 6, 3)).astype(np.uint8)
        image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        aug = iaa.FastSnowyLandscape(
            lightness_threshold=_TwoValueParam(75, 125), lightness_multiplier=2.0
        )

        mask = image_hls[..., 1] < 75
        expected1 = np.copy(image_hls).astype(np.float64)
        expected1[..., 1][mask] *= 2.0
        expected1 = np.clip(np.round(expected1), 0, 255).astype(np.uint8)
        expected1 = cv2.cvtColor(expected1, cv2.COLOR_HLS2RGB)

        mask = image_hls[..., 1] < 125
        expected2 = np.copy(image_hls).astype(np.float64)
        expected2[..., 1][mask] *= 2.0
        expected2 = np.clip(np.round(expected2), 0, 255).astype(np.uint8)
        expected2 = cv2.cvtColor(expected2, cv2.COLOR_HLS2RGB)

        observed = aug.augment_images([image] * 4)

        assert np.array_equal(observed[0], expected1)
        assert np.array_equal(observed[1], expected2)
        assert np.array_equal(observed[2], expected1)
        assert np.array_equal(observed[3], expected2)

    def test_vary_lightness_multiplier(self):
        # test when varying lightness_multiplier between images
        image = np.arange(0, 6 * 6 * 3).reshape((6, 6, 3)).astype(np.uint8)
        image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        aug = iaa.FastSnowyLandscape(
            lightness_threshold=100, lightness_multiplier=_TwoValueParam(1.5, 2.0)
        )

        mask = image_hls[..., 1] < 100
        expected1 = np.copy(image_hls).astype(np.float64)
        expected1[..., 1][mask] *= 1.5
        expected1 = np.clip(np.round(expected1), 0, 255).astype(np.uint8)
        expected1 = cv2.cvtColor(expected1, cv2.COLOR_HLS2RGB)

        mask = image_hls[..., 1] < 100
        expected2 = np.copy(image_hls).astype(np.float64)
        expected2[..., 1][mask] *= 2.0
        expected2 = np.clip(np.round(expected2), 0, 255).astype(np.uint8)
        expected2 = cv2.cvtColor(expected2, cv2.COLOR_HLS2RGB)

        observed = aug.augment_images([image] * 4)

        assert np.array_equal(observed[0], expected1)
        assert np.array_equal(observed[1], expected2)
        assert np.array_equal(observed[2], expected1)
        assert np.array_equal(observed[3], expected2)

    def test_from_colorspace(self):
        # test BGR colorspace
        aug = iaa.FastSnowyLandscape(
            lightness_threshold=100, lightness_multiplier=2.0, from_colorspace="BGR"
        )
        image = np.arange(0, 6 * 6 * 3).reshape((6, 6, 3)).astype(np.uint8)
        image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        mask = image_hls[..., 1] < 100
        expected = np.copy(image_hls).astype(np.float32)
        expected[..., 1][mask] *= 2.0
        expected = np.clip(np.round(expected), 0, 255).astype(np.uint8)
        expected = cv2.cvtColor(expected, cv2.COLOR_HLS2BGR)
        observed = aug.augment_image(image)
        assert np.array_equal(observed, expected)

    def test_zero_sized_axes(self):
        shapes = [(0, 0, 3), (0, 1, 3), (1, 0, 3)]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.FastSnowyLandscape(100, 1.5, from_colorspace="RGB")

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.FastSnowyLandscape(
            lightness_threshold=(50, 150), lightness_multiplier=(1.0, 3.0), seed=1
        )
        runtest_pickleable_uint8_img(aug)


# only a very rough test here currently, because the augmenter is fairly hard
# to test
# TODO add more tests, improve testability
class TestClouds(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _test_very_roughly(cls, nb_channels):
        if nb_channels is None:
            img = np.zeros((100, 100), dtype=np.uint8)
        else:
            img = np.zeros((100, 100, nb_channels), dtype=np.uint8)
        imgs_aug = iaa.Clouds().augment_images([img] * 5)
        assert 20 < np.average(imgs_aug) < 250
        assert np.max(imgs_aug) > 150

        for img_aug in imgs_aug:
            img_aug_f32 = img_aug.astype(np.float32)
            grad_x = img_aug_f32[:, 1:] - img_aug_f32[:, :-1]
            grad_y = img_aug_f32[1:, :] - img_aug_f32[:-1, :]

            assert np.sum(np.abs(grad_x)) > 5 * img.shape[1]
            assert np.sum(np.abs(grad_y)) > 5 * img.shape[0]

    def test_very_roughly_three_channels(self):
        self._test_very_roughly(3)

    def test_very_roughly_one_channel(self):
        self._test_very_roughly(1)

    def test_very_roughly_no_channel(self):
        self._test_very_roughly(None)

    def test_zero_sized_axes(self):
        shapes = [(0, 0), (0, 1), (1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1)]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Clouds()

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [(1, 1, 4), (1, 1, 5), (1, 1, 512), (1, 1, 513)]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Clouds()

                image_aug = aug(image=image)

                assert np.any(image_aug > 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.Clouds(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(20, 20, 3))

    def test_keypoints_not_modified(self):
        # Weather augmenters should not modify keypoints
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(10, 20), ia.Keypoint(30, 40)], shape=(100, 100, 3))
        aug = iaa.Clouds()

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.keypoints[0].x == 10
        assert kpsoi_aug.keypoints[0].y == 20
        assert kpsoi_aug.keypoints[1].x == 30
        assert kpsoi_aug.keypoints[1].y == 40

    def test_heatmaps_not_modified(self):
        # Weather augmenters should not modify heatmaps
        hm_arr = np.zeros((50, 50), dtype=np.float32)
        hm_arr[10:40, 10:40] = 0.8
        hm = ia.HeatmapsOnImage(hm_arr, shape=(100, 100, 3))
        aug = iaa.Clouds()

        hm_aug = aug.augment_heatmaps(hm)

        assert np.allclose(hm_aug.arr_0to1, hm.arr_0to1)

    def test_bounding_boxes_not_modified(self):
        # Weather augmenters should not modify bounding boxes
        bb = ia.BoundingBox(x1=10, y1=20, x2=50, y2=60)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=(100, 100, 3))
        aug = iaa.Clouds()

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert len(bbsoi_aug.bounding_boxes) == 1
        assert bbsoi_aug.bounding_boxes[0].x1 == 10
        assert bbsoi_aug.bounding_boxes[0].y1 == 20
        assert bbsoi_aug.bounding_boxes[0].x2 == 50
        assert bbsoi_aug.bounding_boxes[0].y2 == 60

    def test_polygons_not_modified(self):
        # Weather augmenters should not modify polygons
        poly = ia.Polygon([(10, 10), (50, 10), (50, 50), (10, 50)])
        psoi = ia.PolygonsOnImage([poly], shape=(100, 100, 3))
        aug = iaa.Clouds()

        psoi_aug = aug.augment_polygons(psoi)

        assert len(psoi_aug.polygons) == 1
        assert np.allclose(psoi_aug.polygons[0].exterior, poly.exterior)

    def test_deterministic_produces_same_output(self):
        # Test that deterministic mode produces consistent outputs
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        aug = iaa.Clouds(seed=42)
        aug_det = aug.to_deterministic()

        img_aug1 = aug_det.augment_image(image)
        img_aug2 = aug_det.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)

    def test_seed_produces_reproducible_output(self):
        # Test that same seed produces reproducible output
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        aug1 = iaa.Clouds(seed=123)
        aug2 = iaa.Clouds(seed=123)

        img_aug1 = aug1.augment_image(image)
        img_aug2 = aug2.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)


# only a very rough test here currently, because the augmenter is fairly hard
# to test
class TestFog(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _test_very_roughly(cls, nb_channels):
        if nb_channels is None:
            img = np.zeros((100, 100), dtype=np.uint8)
        else:
            img = np.zeros((100, 100, nb_channels), dtype=np.uint8)
        imgs_aug = iaa.Clouds().augment_images([img] * 5)
        assert 50 < np.average(imgs_aug) < 255
        assert np.max(imgs_aug) > 100

        for img_aug in imgs_aug:
            img_aug_f32 = img_aug.astype(np.float32)
            grad_x = img_aug_f32[:, 1:] - img_aug_f32[:, :-1]
            grad_y = img_aug_f32[1:, :] - img_aug_f32[:-1, :]

            assert np.sum(np.abs(grad_x)) > 1 * img.shape[1]
            assert np.sum(np.abs(grad_y)) > 1 * img.shape[0]

    def test_very_roughly_three_channels(self):
        self._test_very_roughly(3)

    def test_very_roughly_one_channel(self):
        self._test_very_roughly(1)

    def test_very_roughly_no_channel(self):
        self._test_very_roughly(None)

    def test_zero_sized_axes(self):
        shapes = [(0, 0), (0, 1), (1, 0), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1)]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Fog()

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_unusual_channel_numbers(self):
        shapes = [(1, 1, 4), (1, 1, 5), (1, 1, 512), (1, 1, 513)]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Fog()

                image_aug = aug(image=image)

                assert np.any(image_aug > 0)
                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.Fog(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(20, 20, 3))

    def test_keypoints_not_modified(self):
        # Weather augmenters should not modify keypoints
        # Use iaa.weather.Fog to avoid imgcorruptlike.Fog
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(10, 20), ia.Keypoint(30, 40)], shape=(100, 100, 3))
        aug = iaa.weather.Fog()

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.keypoints[0].x == 10
        assert kpsoi_aug.keypoints[0].y == 20

    def test_heatmaps_not_modified(self):
        # Weather augmenters should not modify heatmaps
        hm_arr = np.zeros((50, 50), dtype=np.float32)
        hm_arr[10:40, 10:40] = 0.8
        hm = ia.HeatmapsOnImage(hm_arr, shape=(100, 100, 3))
        aug = iaa.weather.Fog()

        hm_aug = aug.augment_heatmaps(hm)

        assert np.allclose(hm_aug.arr_0to1, hm.arr_0to1)

    def test_bounding_boxes_not_modified(self):
        # Weather augmenters should not modify bounding boxes
        bb = ia.BoundingBox(x1=10, y1=20, x2=50, y2=60)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=(100, 100, 3))
        aug = iaa.weather.Fog()

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert len(bbsoi_aug.bounding_boxes) == 1
        assert bbsoi_aug.bounding_boxes[0].x1 == 10

    def test_deterministic_produces_same_output(self):
        # Test that deterministic mode produces consistent outputs
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        aug = iaa.weather.Fog(seed=42)
        aug_det = aug.to_deterministic()

        img_aug1 = aug_det.augment_image(image)
        img_aug2 = aug_det.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)

    def test_seed_produces_reproducible_output(self):
        # Test that same seed produces reproducible output
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        aug1 = iaa.weather.Fog(seed=123)
        aug2 = iaa.weather.Fog(seed=123)

        img_aug1 = aug1.augment_image(image)
        img_aug2 = aug2.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)


# only a very rough test here currently, because the augmenter is fairly hard
# to test
class TestSnowflakes(unittest.TestCase):
    def setUp(self):
        reseed()

    def _test_very_roughly(self, nb_channels):
        if nb_channels is None:
            img = np.zeros((100, 100), dtype=np.uint8)
        else:
            img = np.zeros((100, 100, nb_channels), dtype=np.uint8)

        imgs_aug = iaa.Snowflakes().augment_images([img] * 5)
        assert 0.01 < np.average(imgs_aug) < 100
        assert np.max(imgs_aug) > 100

        for img_aug in imgs_aug:
            img_aug_f32 = img_aug.astype(np.float32)
            grad_x = img_aug_f32[:, 1:] - img_aug_f32[:, :-1]
            grad_y = img_aug_f32[1:, :] - img_aug_f32[:-1, :]

            assert np.sum(np.abs(grad_x)) > 5 * img.shape[1]
            assert np.sum(np.abs(grad_y)) > 5 * img.shape[0]

        # test density
        imgs_aug_undense = iaa.Snowflakes(density=0.001, density_uniformity=0.99).augment_images(
            [img] * 5
        )
        imgs_aug_dense = iaa.Snowflakes(density=0.1, density_uniformity=0.99).augment_images(
            [img] * 5
        )
        assert np.average(imgs_aug_undense) < np.average(imgs_aug_dense)

        # test density_uniformity
        imgs_aug_ununiform = iaa.Snowflakes(density=0.4, density_uniformity=0.1).augment_images(
            [img] * 30
        )
        imgs_aug_uniform = iaa.Snowflakes(density=0.4, density_uniformity=0.9).augment_images(
            [img] * 30
        )

        ununiform_uniformity = np.average(
            [self._measure_uniformity(img_aug) for img_aug in imgs_aug_ununiform]
        )
        uniform_uniformity = np.average(
            [self._measure_uniformity(img_aug) for img_aug in imgs_aug_uniform]
        )

        assert ununiform_uniformity < uniform_uniformity

    def test_very_roughly_three_channels(self):
        self._test_very_roughly(3)

    def test_very_roughly_one_channel(self):
        self._test_very_roughly(1)

    def test_very_roughly_no_channels(self):
        self._test_very_roughly(None)

    def test_zero_sized_axes(self):
        shapes = [(0, 0, 3), (0, 1, 3), (1, 0, 3)]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Snowflakes()

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.Snowflakes(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(20, 20, 3))

    def test_keypoints_not_modified(self):
        # Weather augmenters should not modify keypoints
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(10, 20), ia.Keypoint(30, 40)], shape=(100, 100, 3))
        aug = iaa.Snowflakes()

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.keypoints[0].x == 10
        assert kpsoi_aug.keypoints[0].y == 20

    def test_heatmaps_not_modified(self):
        # Weather augmenters should not modify heatmaps
        hm_arr = np.zeros((50, 50), dtype=np.float32)
        hm_arr[10:40, 10:40] = 0.8
        hm = ia.HeatmapsOnImage(hm_arr, shape=(100, 100, 3))
        aug = iaa.Snowflakes()

        hm_aug = aug.augment_heatmaps(hm)

        assert np.allclose(hm_aug.arr_0to1, hm.arr_0to1)

    def test_bounding_boxes_not_modified(self):
        # Weather augmenters should not modify bounding boxes
        bb = ia.BoundingBox(x1=10, y1=20, x2=50, y2=60)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=(100, 100, 3))
        aug = iaa.Snowflakes()

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert len(bbsoi_aug.bounding_boxes) == 1
        assert bbsoi_aug.bounding_boxes[0].x1 == 10

    def test_deterministic_produces_same_output(self):
        # Test that deterministic mode produces consistent outputs
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        aug = iaa.Snowflakes(seed=42)
        aug_det = aug.to_deterministic()

        img_aug1 = aug_det.augment_image(image)
        img_aug2 = aug_det.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)

    def test_seed_produces_reproducible_output(self):
        # Test that same seed produces reproducible output
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        aug1 = iaa.Snowflakes(seed=123)
        aug2 = iaa.Snowflakes(seed=123)

        img_aug1 = aug1.augment_image(image)
        img_aug2 = aug2.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)

    @classmethod
    def _measure_uniformity(cls, image, patch_size=5, n_patches=50):
        pshalf = (patch_size - 1) // 2
        image_f32 = image.astype(np.float32)
        grad_x = image_f32[:, 1:] - image_f32[:, :-1]
        grad_y = image_f32[1:, :] - image_f32[:-1, :]
        grad = np.abs(grad_x[1:, :] + grad_y[:, 1:])
        points_y = np.random.randint(0, image.shape[0], size=(n_patches,))
        points_x = np.random.randint(0, image.shape[0], size=(n_patches,))
        stds = []
        for y, x in zip(points_y, points_x, strict=False):
            bb = ia.BoundingBox(x1=x - pshalf, y1=y - pshalf, x2=x + pshalf, y2=y + pshalf)
            patch = bb.extract_from_image(grad)
            stds.append(np.std(patch))
        return 1 / (1 + np.std(stds))


class TestSnowflakesLayer(unittest.TestCase):
    def setUp(self):
        reseed()

    def test_large_snowflakes_size(self):
        # Test for PR #471
        # Snowflakes size is achieved via downscaling. Large values for
        # snowflakes_size lead to more downscaling. Hence, values close to 1.0
        # incur risk that the image is downscaled to (0, 0) or similar values.
        aug = iaa.SnowflakesLayer(
            density=0.95,
            density_uniformity=0.5,
            flake_size=1.0,
            flake_size_uniformity=0.5,
            angle=0.0,
            speed=0.5,
            blur_sigma_fraction=0.001,
        )

        nb_seen = 0
        for _ in np.arange(50):
            image = np.zeros((16, 16, 3), dtype=np.uint8)

            image_aug = aug.augment_image(image)

            assert np.std(image_aug) < 1
            if np.average(image_aug) > 128:
                nb_seen += 1
        assert nb_seen > 30  # usually around 45


# only a very rough test here currently, because the augmenter is fairly hard
# to test
# TODO add more tests, improve testability
class TestRain(unittest.TestCase):
    def setUp(self):
        reseed()

    @classmethod
    def _test_very_roughly(cls, nb_channels):
        if nb_channels is None:
            img = np.zeros((100, 100), dtype=np.uint8)
        else:
            img = np.zeros((100, 100, nb_channels), dtype=np.uint8)

        imgs_aug = iaa.Rain()(images=[img] * 5)
        assert 5 < np.average(imgs_aug) < 200
        assert np.max(imgs_aug) > 70

        for img_aug in imgs_aug:
            img_aug_f32 = img_aug.astype(np.float32)
            grad_x = img_aug_f32[:, 1:] - img_aug_f32[:, :-1]
            grad_y = img_aug_f32[1:, :] - img_aug_f32[:-1, :]

            assert np.sum(np.abs(grad_x)) > 10 * img.shape[1]
            assert np.sum(np.abs(grad_y)) > 10 * img.shape[0]

    def test_very_roughly_three_channels(self):
        self._test_very_roughly(3)

    def test_very_roughly_one_channel(self):
        self._test_very_roughly(1)

    def test_very_roughly_no_channels(self):
        self._test_very_roughly(None)

    def test_zero_sized_axes(self):
        shapes = [(0, 0, 3), (0, 1, 3), (1, 0, 3)]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)
                aug = iaa.Rain()

                image_aug = aug(image=image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape

    def test_pickleable(self):
        aug = iaa.Rain(seed=1)
        runtest_pickleable_uint8_img(aug, iterations=3, shape=(20, 20, 3))

    def test_keypoints_not_modified(self):
        # Weather augmenters should not modify keypoints
        kpsoi = ia.KeypointsOnImage([ia.Keypoint(10, 20), ia.Keypoint(30, 40)], shape=(100, 100, 3))
        aug = iaa.Rain()

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.keypoints[0].x == 10
        assert kpsoi_aug.keypoints[0].y == 20

    def test_heatmaps_not_modified(self):
        # Weather augmenters should not modify heatmaps
        hm_arr = np.zeros((50, 50), dtype=np.float32)
        hm_arr[10:40, 10:40] = 0.8
        hm = ia.HeatmapsOnImage(hm_arr, shape=(100, 100, 3))
        aug = iaa.Rain()

        hm_aug = aug.augment_heatmaps(hm)

        assert np.allclose(hm_aug.arr_0to1, hm.arr_0to1)

    def test_bounding_boxes_not_modified(self):
        # Weather augmenters should not modify bounding boxes
        bb = ia.BoundingBox(x1=10, y1=20, x2=50, y2=60)
        bbsoi = ia.BoundingBoxesOnImage([bb], shape=(100, 100, 3))
        aug = iaa.Rain()

        bbsoi_aug = aug.augment_bounding_boxes(bbsoi)

        assert len(bbsoi_aug.bounding_boxes) == 1
        assert bbsoi_aug.bounding_boxes[0].x1 == 10

    def test_deterministic_produces_same_output(self):
        # Test that deterministic mode produces consistent outputs
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        aug = iaa.Rain(seed=42)
        aug_det = aug.to_deterministic()

        img_aug1 = aug_det.augment_image(image)
        img_aug2 = aug_det.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)

    def test_seed_produces_reproducible_output(self):
        # Test that same seed produces reproducible output
        image = np.zeros((50, 50, 3), dtype=np.uint8)

        aug1 = iaa.Rain(seed=123)
        aug2 = iaa.Rain(seed=123)

        img_aug1 = aug1.augment_image(image)
        img_aug2 = aug2.augment_image(image)

        assert np.array_equal(img_aug1, img_aug2)


class TestCloudLayer(unittest.TestCase):
    def setUp(self):
        reseed()

    def _create_cloud_layer(self, seed=None):
        """Helper to create a CloudLayer with typical parameters."""
        return iaa.weather.CloudLayer(
            intensity_mean=(196, 255),
            intensity_freq_exponent=(-2.5, -2.0),
            intensity_coarse_scale=10,
            alpha_min=0,
            alpha_multiplier=(0.25, 0.75),
            alpha_size_px_max=(2, 8),
            alpha_freq_exponent=(-2.5, -2.0),
            sparsity=(0.8, 1.0),
            density_multiplier=(0.5, 1.0),
            seed=seed,
        )

    def test_basic_augmentation(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        aug = self._create_cloud_layer()

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == image.shape
        # Should add some white cloud pixels
        assert np.mean(image_aug) > 0

    def test_rgb_image(self):
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        aug = self._create_cloud_layer()

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == image.shape

    def test_grayscale_image(self):
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        aug = self._create_cloud_layer()

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == image.shape

    def test_float32_image(self):
        image = np.random.rand(50, 50, 3).astype(np.float32)
        aug = self._create_cloud_layer()

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == "float32"
        assert image_aug.shape == image.shape

    def test_multiple_images(self):
        images = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        aug = self._create_cloud_layer()

        images_aug = aug.augment_images(images)

        assert len(images_aug) == len(images)
        for img_aug in images_aug:
            assert img_aug.shape == images[0].shape
            assert img_aug.dtype.name == "uint8"

    def test_determinism(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        aug = self._create_cloud_layer(seed=42)

        image_aug1 = aug.augment_image(image)
        aug = self._create_cloud_layer(seed=42)
        image_aug2 = aug.augment_image(image)

        assert np.array_equal(image_aug1, image_aug2)

    def test_get_parameters(self):
        aug = self._create_cloud_layer()
        params = aug.get_parameters()

        assert len(params) == 9
        assert params[0] is aug.intensity_mean
        assert params[1] is aug.alpha_min
        assert params[2] is aug.alpha_multiplier

    def test_keypoints_not_modified(self):
        kpsoi = ia.KeypointsOnImage(
            [ia.Keypoint(10, 20), ia.Keypoint(30, 40)], shape=(100, 100, 3)
        )
        aug = self._create_cloud_layer()

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.keypoints[0].x == 10
        assert kpsoi_aug.keypoints[0].y == 20


class TestRainLayer(unittest.TestCase):
    def setUp(self):
        reseed()

    def _create_rain_layer(self, seed=None):
        """Helper to create a RainLayer with typical parameters."""
        return iaa.weather.RainLayer(
            density=(0.03, 0.14),
            density_uniformity=(0.8, 1.0),
            drop_size=(0.01, 0.02),
            drop_size_uniformity=(0.2, 0.5),
            angle=(-15, 15),
            speed=(0.04, 0.20),
            blur_sigma_fraction=(0.001, 0.001),
            seed=seed,
        )

    def test_basic_augmentation(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        aug = self._create_rain_layer()

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == image.shape

    def test_rgb_image(self):
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        aug = self._create_rain_layer()

        image_aug = aug.augment_image(image)

        assert image_aug.dtype.name == "uint8"
        assert image_aug.shape == image.shape

    def test_multiple_images(self):
        images = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        aug = self._create_rain_layer()

        images_aug = aug.augment_images(images)

        assert len(images_aug) == len(images)
        for img_aug in images_aug:
            assert img_aug.shape == images[0].shape
            assert img_aug.dtype.name == "uint8"

    def test_determinism(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        aug = self._create_rain_layer(seed=42)

        image_aug1 = aug.augment_image(image)
        aug = self._create_rain_layer(seed=42)
        image_aug2 = aug.augment_image(image)

        assert np.array_equal(image_aug1, image_aug2)

    def test_keypoints_not_modified(self):
        kpsoi = ia.KeypointsOnImage(
            [ia.Keypoint(10, 20), ia.Keypoint(30, 40)], shape=(100, 100, 3)
        )
        aug = self._create_rain_layer()

        kpsoi_aug = aug.augment_keypoints(kpsoi)

        assert len(kpsoi_aug.keypoints) == 2
        assert kpsoi_aug.keypoints[0].x == 10
        assert kpsoi_aug.keypoints[0].y == 20
