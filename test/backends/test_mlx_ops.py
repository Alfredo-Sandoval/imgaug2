import unittest

import cv2
import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False
    mx = None


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsGaussianBlur(unittest.TestCase):
    def test_matches_cv2_uint8(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        sigma = 3.0

        observed = mlx_blur.gaussian_blur(image, sigma=sigma)

        ksize = 9  # matches imgaug2.augmenters.blur._compute_gaussian_blur_ksize(3.0)
        expected = cv2.GaussianBlur(
            image,
            (ksize, ksize),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT_101,
        )

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert observed.dtype == np.uint8
        assert diff.max() <= 2

    def test_preserves_type_for_mlx_inputs(self):
        from imgaug2.mlx import blur as mlx_blur

        image = mx.ones((16, 16, 3), dtype=mx.float32)

        out = mlx_blur.gaussian_blur(image, sigma=1.0)

        assert isinstance(out, mx.array)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsMedianBlur(unittest.TestCase):
    def test_matches_cv2_uint8(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        observed = mlx_blur.median_blur(image, ksize=3)
        expected = cv2.medianBlur(image, 3)

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert observed.dtype == np.uint8
        assert diff.max() <= 2


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsMotionBlur(unittest.TestCase):
    def test_matches_cv2_filter2d_uint8(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        k = 9
        angle = 30.0
        observed = mlx_blur.motion_blur(image, k=k, angle=angle)

        center = (k - 1) / 2.0
        m = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel_line = np.zeros((k, k), dtype=np.float32)
        cv2.line(kernel_line, (0, int(center)), (k - 1, int(center)), 1.0)
        kernel = cv2.warpAffine(kernel_line, m, (k, k))
        s = kernel.sum()
        if s > 0:
            kernel /= s

        expected = cv2.filter2D(image, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101)

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert observed.dtype == np.uint8
        assert diff.max() <= 3


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsDownscale(unittest.TestCase):
    def test_scale_1_is_identity(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        observed = mlx_blur.downscale(image, scale=1.0)
        np.testing.assert_array_equal(observed, image)

    def test_downscale_changes_image(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        observed = mlx_blur.downscale(image, scale=0.5)
        assert observed.shape == image.shape
        assert observed.dtype == np.uint8
        assert np.mean(np.abs(observed.astype(np.int16) - image.astype(np.int16))) >= 0.5


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsJpegCompression(unittest.TestCase):
    def test_preserves_shape_and_dtype(self):
        from imgaug2.mlx import jpeg_compression

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        out = jpeg_compression(image, quality=30)
        assert isinstance(out, np.ndarray)
        assert out.shape == image.shape
        assert out.dtype == np.uint8

    def test_lower_quality_has_more_error(self):
        from imgaug2.mlx import jpeg_compression

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)

        hi = jpeg_compression(image, quality=90)
        lo = jpeg_compression(image, quality=20)

        err_hi = float(np.mean(np.abs(hi.astype(np.int16) - image.astype(np.int16))))
        err_lo = float(np.mean(np.abs(lo.astype(np.int16) - image.astype(np.int16))))
        assert err_lo >= err_hi


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsEasyWins(unittest.TestCase):
    def test_fliplr_matches_numpy(self):
        from imgaug2.mlx import flip as mlx_flip

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        observed = mlx_flip.fliplr(image)
        expected = image[:, ::-1, :]

        assert isinstance(observed, np.ndarray)
        assert observed.dtype == np.uint8
        np.testing.assert_array_equal(observed, expected)

    def test_flipud_matches_numpy(self):
        from imgaug2.mlx import flip as mlx_flip

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        observed = mlx_flip.flipud(image)
        expected = image[::-1, :, :]

        np.testing.assert_array_equal(observed, expected)

    def test_rot90_matches_numpy(self):
        from imgaug2.mlx import flip as mlx_flip

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(16, 20, 3), dtype=np.uint8)

        for k in [1, 2, 3]:
            observed = mlx_flip.rot90(image, k=k)
            expected = np.rot90(image, k=k)
            np.testing.assert_array_equal(observed, expected)

    def test_flip_points_matches_expected(self):
        from imgaug2.mlx import flip as mlx_flip

        points = np.array([[0.0, 0.0], [10.0, 5.0]], dtype=np.float32)
        shape = (12, 20)

        lr = mlx_flip.fliplr_points(points, shape)
        ud = mlx_flip.flipud_points(points, shape)

        expected_lr = np.array([[20.0, 0.0], [10.0, 5.0]], dtype=np.float32)
        expected_ud = np.array([[0.0, 12.0], [10.0, 7.0]], dtype=np.float32)

        np.testing.assert_allclose(lr, expected_lr, atol=1e-6)
        np.testing.assert_allclose(ud, expected_ud, atol=1e-6)

    def test_rot90_points_matches_expected(self):
        from imgaug2.mlx import flip as mlx_flip

        points = np.array([[2.0, 3.0], [7.0, 1.0]], dtype=np.float32)
        shape = (10, 20)

        k1 = mlx_flip.rot90_points(points, shape, k=1)
        expected_k1 = np.array([[10.0 - 3.0, 2.0], [10.0 - 1.0, 7.0]], dtype=np.float32)

        k2 = mlx_flip.rot90_points(points, shape, k=2)
        expected_k2 = np.array([[20.0 - 2.0, 10.0 - 3.0], [20.0 - 7.0, 10.0 - 1.0]], dtype=np.float32)

        np.testing.assert_allclose(k1, expected_k1, atol=1e-6)
        np.testing.assert_allclose(k2, expected_k2, atol=1e-6)

    def test_pillike_warp_affine_matches_pil(self):
        import imgaug2.augmenters.pillike as pl

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)

        params = dict(
            scale_x=1.0,
            scale_y=1.0,
            translate_x_px=3.0,
            translate_y_px=-2.0,
            rotate_deg=0.0,
            shear_x_deg=0.0,
            shear_y_deg=0.0,
            fillcolor=(10, 20, 30),
            center=(0.5, 0.5),
        )

        expected = pl.warp_affine(image, **params)
        observed = pl.warp_affine(mx.array(image), **params)
        observed_np = np.array(observed)

        np.testing.assert_array_equal(observed_np, expected)

    def test_invert_and_solarize_match_numpy(self):
        from imgaug2.mlx import pointwise as mlx_pointwise

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        inv = mlx_pointwise.invert(image)
        expected_inv = 255 - image
        np.testing.assert_array_equal(inv, expected_inv)

        sol = mlx_pointwise.solarize(image, threshold=128)
        expected_sol = np.where(image >= 128, 255 - image, image).astype(np.uint8)
        np.testing.assert_array_equal(sol, expected_sol)

    def test_gamma_contrast_matches_reference_uint8(self):
        from imgaug2.mlx import pointwise as mlx_pointwise

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        gamma = 1.7
        observed = mlx_pointwise.gamma_contrast(image, gamma)

        x01 = image.astype(np.float32) / 255.0
        expected = np.clip((x01**gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
        np.testing.assert_array_equal(observed, expected)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsPooling(unittest.TestCase):
    def test_avg_pool_matches_imgaug_uint8(self):
        import imgaug2.imgaug as ia
        from imgaug2.mlx import pooling as mlx_pooling

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(37, 41, 3), dtype=np.uint8)

        observed = mlx_pooling.avg_pool(image, (4, 5))
        expected = ia.avg_pool(image, (4, 5))

        assert observed.dtype == np.uint8
        assert observed.shape == expected.shape

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert diff.max() <= 2

    def test_max_pool_matches_imgaug_uint8(self):
        import imgaug2.imgaug as ia
        from imgaug2.mlx import pooling as mlx_pooling

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(33, 47, 3), dtype=np.uint8)

        observed = mlx_pooling.max_pool(image, (3, 4))
        expected = ia.max_pool(image, (3, 4))

        np.testing.assert_array_equal(observed, expected)

    def test_min_pool_matches_imgaug_uint8(self):
        import imgaug2.imgaug as ia
        from imgaug2.mlx import pooling as mlx_pooling

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(33, 47, 3), dtype=np.uint8)

        observed = mlx_pooling.min_pool(image, (3, 4))
        expected = ia.min_pool(image, (3, 4))

        np.testing.assert_array_equal(observed, expected)

    def test_preserves_type_for_mlx_inputs(self):
        from imgaug2.mlx import pooling as mlx_pooling

        image = mx.ones((16, 16, 3), dtype=mx.float32)
        out = mlx_pooling.avg_pool(image, 2, preserve_dtype=False)

        assert isinstance(out, mx.array)
        assert out.shape == (8, 8, 3)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsGeometrySanity(unittest.TestCase):
    def test_affine_identity_matches_input_uint8(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        mat = np.eye(3, dtype=np.float32)

        observed = mlx_geom.affine_transform(image, mat, order=1, mode="reflect_101")
        np.testing.assert_array_equal(observed, image)

    def test_perspective_identity_matches_input_uint8(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
        mat = np.eye(3, dtype=np.float32)

        observed = mlx_geom.perspective_transform(image, mat, order=1, mode="reflect_101")
        np.testing.assert_array_equal(observed, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsColor(unittest.TestCase):
    def test_rgb_shift_adds_correctly(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        observed = mlx_color.rgb_shift(image, r_shift=10, g_shift=-10, b_shift=5)

        expected = image.astype(np.float32) + np.array([10, -10, 5], dtype=np.float32)
        expected = np.clip(expected, 0, 255).astype(np.uint8)

        np.testing.assert_array_equal(observed, expected)

    def test_channel_shuffle_with_explicit_order(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        # BGR to RGB
        observed = mlx_color.channel_shuffle(image, order=[2, 1, 0])
        expected = image[:, :, ::-1]

        np.testing.assert_array_equal(observed, expected)

    def test_channel_shuffle_random_is_permutation(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        observed = mlx_color.channel_shuffle(image, seed=42)

        # Check that result contains the same channels, just reordered
        for c in range(3):
            original_channel = image[:, :, c]
            found = any(
                np.array_equal(observed[:, :, i], original_channel) for i in range(3)
            )
            assert found, f"Channel {c} not found in output"

    def test_normalize_and_denormalize_roundtrip(self):
        from imgaug2.mlx import color as mlx_color

        image = np.random.rand(32, 48, 3).astype(np.float32)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalized = mlx_color.normalize(image, mean=mean, std=std)
        denormalized = mlx_color.denormalize(normalized, mean=mean, std=std)

        np.testing.assert_allclose(denormalized, image, rtol=1e-5, atol=1e-5)

    def test_to_float_from_float_roundtrip(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        float_img = mlx_color.to_float(image)
        assert float_img.dtype == np.float32
        assert float_img.min() >= 0.0
        assert float_img.max() <= 1.0

        back = mlx_color.from_float(float_img)
        assert back.dtype == np.uint8
        np.testing.assert_array_equal(back, image)

    def test_grayscale_shape_preserved(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        gray = mlx_color.grayscale(image)

        assert gray.shape == image.shape
        assert gray.dtype == np.uint8
        # All channels should be equal
        np.testing.assert_array_equal(gray[:, :, 0], gray[:, :, 1])
        np.testing.assert_array_equal(gray[:, :, 1], gray[:, :, 2])


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsCrop(unittest.TestCase):
    def test_center_crop_correct_size(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(100, 120, 3), dtype=np.uint8)

        cropped = mlx_crop.center_crop(image, height=50, width=60)

        assert cropped.shape == (50, 60, 3)
        assert cropped.dtype == np.uint8
        # Verify it's actually the center
        expected = image[25:75, 30:90, :]
        np.testing.assert_array_equal(cropped, expected)

    def test_center_crop_batch(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(4, 100, 120, 3), dtype=np.uint8)

        cropped = mlx_crop.center_crop(batch, height=50, width=60)

        assert cropped.shape == (4, 50, 60, 3)
        expected = batch[:, 25:75, 30:90, :]
        np.testing.assert_array_equal(cropped, expected)

    def test_random_crop_correct_size(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(100, 120, 3), dtype=np.uint8)

        cropped = mlx_crop.random_crop(image, height=50, width=60, seed=42)

        assert cropped.shape == (50, 60, 3)
        assert cropped.dtype == np.uint8

    def test_random_crop_reproducible(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(100, 120, 3), dtype=np.uint8)

        crop1 = mlx_crop.random_crop(image, height=50, width=60, seed=42)
        crop2 = mlx_crop.random_crop(image, height=50, width=60, seed=42)

        np.testing.assert_array_equal(crop1, crop2)

    def test_pad_constant(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        padded = mlx_crop.pad(image, pad_top=5, pad_bottom=10, pad_left=3, pad_right=7, value=128)

        assert padded.shape == (47, 58, 3)
        # Check that center is original
        np.testing.assert_array_equal(padded[5:37, 3:51, :], image)
        # Check padding values
        assert padded[0, 0, 0] == 128

    def test_pad_edge(self):
        from imgaug2.mlx import crop as mlx_crop

        image = np.arange(12).reshape(3, 4, 1).astype(np.uint8)

        padded = mlx_crop.pad(image, pad_top=1, pad_bottom=1, pad_left=1, pad_right=1, mode="edge")

        assert padded.shape == (5, 6, 1)
        # Top-left corner should be edge value
        assert padded[0, 0, 0] == image[0, 0, 0]
        # Bottom-right corner
        assert padded[-1, -1, 0] == image[-1, -1, 0]

    def test_pad_if_needed_does_nothing_when_large_enough(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)

        result = mlx_crop.pad_if_needed(image, min_height=50, min_width=50)

        # Should return the same array since it's already large enough
        assert result is image

    def test_pad_if_needed_pads_small_image(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)

        result = mlx_crop.pad_if_needed(image, min_height=50, min_width=60, position="center")

        assert result.shape == (50, 60, 3)

    def test_random_resized_crop_output_size(self):
        from imgaug2.mlx import crop as mlx_crop

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

        cropped = mlx_crop.random_resized_crop(
            image, height=224, width=224, scale=(0.08, 1.0), seed=42
        )

        assert cropped.shape == (224, 224, 3)
        assert cropped.dtype == np.uint8


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsNoise(unittest.TestCase):
    def test_multiplicative_noise_changes_image(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_noise.multiplicative_noise(image, scale=0.1, seed=42)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should be different from original
        assert not np.array_equal(result, image)

    def test_grid_dropout_creates_grid_pattern(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(0)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = mlx_noise.grid_dropout(image, ratio=0.5, grid_size=(4, 4), seed=42)

        assert result.shape == image.shape
        # Some pixels should be zeroed
        assert result.min() == 0
        # Some should remain
        assert result.max() > 0


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsHSV(unittest.TestCase):
    def test_rgb_hsv_roundtrip(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.random((32, 48, 3)).astype(np.float32)

        hsv = mlx_color.rgb_to_hsv(image)
        rgb_back = mlx_color.hsv_to_rgb(hsv)

        np.testing.assert_allclose(rgb_back, image, rtol=1e-4, atol=1e-4)

    def test_hue_saturation_value_adjusts(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        # Increase saturation
        result = mlx_color.hue_saturation_value(
            image, hue_shift=0, saturation_scale=1.5, value_scale=1.0
        )

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Result should be different
        assert not np.array_equal(result, image)

    def test_hue_shift_changes_colors(self):
        from imgaug2.mlx import color as mlx_color

        # Create a pure red image
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:, :, 0] = 255  # Red channel

        # Shift hue by 120 degrees (should become green-ish)
        result = mlx_color.hue_saturation_value(
            image, hue_shift=120, saturation_scale=1.0, value_scale=1.0
        )

        assert result.shape == image.shape
        # Red channel should decrease, green should increase
        assert result[:, :, 0].mean() < 255
        assert result[:, :, 1].mean() > 0


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsSharpen(unittest.TestCase):
    def test_sharpen_enhances_edges(self):
        from imgaug2.mlx.sharpen import sharpen

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        result = sharpen(image, alpha=1.0, lightness=1.0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_emboss_creates_relief_effect(self):
        from imgaug2.mlx.sharpen import emboss

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        result = emboss(image, alpha=1.0, strength=1.0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Emboss adds 128, so values should be shifted
        assert not np.array_equal(result, image)

    def test_unsharp_mask_sharpens(self):
        from imgaug2.mlx.sharpen import unsharp_mask

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        result = unsharp_mask(image, sigma=1.0, strength=1.0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should be different from original
        assert not np.array_equal(result, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsEqualize(unittest.TestCase):
    def test_equalize_output_shape_and_dtype(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.equalize(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_equalize_spreads_histogram(self):
        from imgaug2.mlx import color as mlx_color

        # Create a low-contrast image (values clustered in middle)
        rng = np.random.default_rng(0)
        image = rng.integers(100, 150, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.equalize(image)

        # After equalization, dynamic range should increase
        assert result.max() > image.max() or result.min() < image.min()

    def test_equalize_batch(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(2, 32, 48, 3), dtype=np.uint8)

        result = mlx_color.equalize(batch)

        assert result.shape == batch.shape
        assert result.dtype == np.uint8


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsChannelDropout(unittest.TestCase):
    def test_channel_dropout_zeros_channel(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        # Drop red channel
        result = mlx_color.channel_dropout(image, channel_idx=0, fill_value=0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Red channel should be all zeros
        np.testing.assert_array_equal(result[:, :, 0], 0)
        # Other channels should be unchanged
        np.testing.assert_array_equal(result[:, :, 1], image[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], image[:, :, 2])

    def test_channel_dropout_multiple_channels(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        # Drop red and blue channels
        result = mlx_color.channel_dropout(image, channel_idx=[0, 2], fill_value=0)

        np.testing.assert_array_equal(result[:, :, 0], 0)
        np.testing.assert_array_equal(result[:, :, 1], image[:, :, 1])
        np.testing.assert_array_equal(result[:, :, 2], 0)

    def test_channel_dropout_random(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        # Random channel dropout
        result = mlx_color.channel_dropout(image, seed=42)

        assert result.shape == image.shape
        # At least one channel should be zeroed
        zero_channels = sum(
            np.all(result[:, :, c] == 0) for c in range(3)
        )
        assert zero_channels >= 1

    def test_channel_dropout_custom_fill_value(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.channel_dropout(image, channel_idx=1, fill_value=128)

        np.testing.assert_array_equal(result[:, :, 1], 128)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsCLAHE(unittest.TestCase):
    def test_clahe_output_shape_and_dtype(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        # Use small image for speed (CLAHE is slow)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        result = mlx_color.clahe(image, clip_limit=2.0, tile_grid_size=(2, 2))

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_clahe_enhances_contrast(self):
        from imgaug2.mlx import color as mlx_color

        # Create low-contrast image
        rng = np.random.default_rng(0)
        image = rng.integers(100, 150, size=(16, 16, 3), dtype=np.uint8)

        result = mlx_color.clahe(image, clip_limit=4.0, tile_grid_size=(2, 2))

        # After CLAHE, range should be broader
        result_range = result.max() - result.min()
        input_range = image.max() - image.min()
        assert result_range >= input_range


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsPosterize(unittest.TestCase):
    def test_posterize_reduces_colors(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.posterize(image, bits=4)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # With 4 bits, values should be multiples of 16
        unique_vals = np.unique(result)
        assert all(v % 16 == 0 for v in unique_vals)

    def test_posterize_bits_1(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.posterize(image, bits=1)

        # With 1 bit, only 0 and 128 possible
        unique_vals = np.unique(result)
        assert all(v in [0, 128] for v in unique_vals)

    def test_posterize_bits_8_is_identity(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.posterize(image, bits=8)

        # 8 bits should preserve the image
        np.testing.assert_array_equal(result, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsAutocontrast(unittest.TestCase):
    def test_autocontrast_stretches_histogram(self):
        from imgaug2.mlx import color as mlx_color

        # Create low-contrast image
        rng = np.random.default_rng(0)
        image = rng.integers(100, 150, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.autocontrast(image)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should stretch to full range
        assert result.min() < image.min() or result.max() > image.max()

    def test_autocontrast_with_cutoff(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.autocontrast(image, cutoff=5.0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsColorJitter(unittest.TestCase):
    def test_color_jitter_brightness(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.color_jitter(image, brightness=0.5)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Brighter image should have higher mean
        assert result.mean() > image.mean()

    def test_color_jitter_contrast(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.color_jitter(image, contrast=0.5)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_color_jitter_saturation(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(50, 200, size=(32, 48, 3), dtype=np.uint8)

        result = mlx_color.color_jitter(image, saturation=0.5)

        assert result.shape == image.shape
        assert result.dtype == np.uint8


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsCutout(unittest.TestCase):
    def test_cutout_creates_holes(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(0)
        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = mlx_noise.cutout(
            image, num_holes=1, hole_height=16, hole_width=16, fill_value=0, seed=42
        )

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should have some zeros from the cutout
        assert result.min() == 0
        # Should have some original values
        assert result.max() == 128

    def test_cutout_multiple_holes(self):
        from imgaug2.mlx import noise as mlx_noise

        image = np.ones((64, 64, 3), dtype=np.uint8) * 255

        result = mlx_noise.cutout(
            image, num_holes=3, hole_height=8, hole_width=8, fill_value=0, seed=42
        )

        # Count zeros
        zero_pixels = np.sum(result == 0)
        assert zero_pixels > 0

    def test_cutout_custom_fill_value(self):
        from imgaug2.mlx import noise as mlx_noise

        image = np.zeros((64, 64, 3), dtype=np.uint8)

        result = mlx_noise.cutout(
            image, num_holes=1, hole_height=32, hole_width=32, fill_value=128, seed=42
        )

        # Should have filled region with 128
        assert 128 in result


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsRandomErasing(unittest.TestCase):
    def test_random_erasing_with_probability_1(self):
        from imgaug2.mlx import noise as mlx_noise

        image = np.ones((64, 64, 3), dtype=np.uint8) * 128

        result = mlx_noise.random_erasing(
            image, p=1.0, scale=(0.1, 0.3), fill_value=0, seed=42
        )

        assert result.shape == image.shape
        # Should have erased region
        assert result.min() == 0

    def test_random_erasing_with_probability_0(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result = mlx_noise.random_erasing(image, p=0.0, seed=42)

        # Should be unchanged
        np.testing.assert_array_equal(result, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsMedianBlur(unittest.TestCase):
    def test_median_blur_removes_noise(self):
        from imgaug2.mlx import blur as mlx_blur

        # Create image with salt-and-pepper noise
        image = np.ones((32, 32, 3), dtype=np.uint8) * 128
        rng = np.random.default_rng(0)
        noise_mask = rng.random((32, 32)) < 0.1
        image[noise_mask] = 255

        result = mlx_blur.median_blur(image, ksize=3)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Median filter should smooth out the noise
        # Variance should be reduced
        assert result.var() < image.var()

    def test_median_blur_batch(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)

        result = mlx_blur.median_blur(batch, ksize=3)

        assert result.shape == batch.shape
        assert result.dtype == np.uint8

    def test_median_blur_preserves_mlx_type(self):
        from imgaug2.mlx import blur as mlx_blur

        image = mx.ones((16, 16, 3), dtype=mx.float32)

        result = mlx_blur.median_blur(image, ksize=3)

        assert isinstance(result, mx.array)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsMotionBlur(unittest.TestCase):
    def test_motion_blur_blurs_image(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_blur.motion_blur(image, k=5, angle=0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should be different from original
        assert not np.array_equal(result, image)

    def test_motion_blur_different_angles(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result_0 = mlx_blur.motion_blur(image, k=5, angle=0)
        result_45 = mlx_blur.motion_blur(image, k=5, angle=45)
        result_90 = mlx_blur.motion_blur(image, k=5, angle=90)

        # Different angles should produce different results
        assert not np.array_equal(result_0, result_45)
        assert not np.array_equal(result_0, result_90)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsDownscale(unittest.TestCase):
    def test_downscale_produces_artifacts(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result = mlx_blur.downscale(image, scale=0.25)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should be different from original (lower quality)
        assert not np.array_equal(result, image)

    def test_downscale_scale_1_returns_similar(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result = mlx_blur.downscale(image, scale=1.0)

        # Scale 1.0 should return nearly identical image
        assert result.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsMorphology(unittest.TestCase):
    def test_erosion_shrinks_bright_regions(self):
        from imgaug2.mlx import morphology as mlx_morph

        # Create image with white square on black background
        image = np.zeros((32, 32, 1), dtype=np.uint8)
        image[10:22, 10:22] = 255

        result = mlx_morph.erosion(image, ksize=3)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # White region should be smaller (less white pixels)
        assert result.sum() < image.sum()

    def test_dilation_expands_bright_regions(self):
        from imgaug2.mlx import morphology as mlx_morph

        # Create image with white square on black background
        image = np.zeros((32, 32, 1), dtype=np.uint8)
        image[10:22, 10:22] = 255

        result = mlx_morph.dilation(image, ksize=3)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # White region should be larger (more white pixels)
        assert result.sum() > image.sum()

    def test_opening_removes_small_spots(self):
        from imgaug2.mlx import morphology as mlx_morph

        # Create image with large white square and small noise spots
        image = np.zeros((32, 32, 1), dtype=np.uint8)
        image[10:22, 10:22] = 255
        image[5, 5] = 255  # Small spot
        image[28, 28] = 255  # Small spot

        result = mlx_morph.opening(image, ksize=3)

        assert result.shape == image.shape
        # Opening should remove small spots but keep large square
        assert result[5, 5, 0] == 0  # Small spot removed
        assert result[15, 15, 0] == 255  # Large square preserved

    def test_closing_fills_small_holes(self):
        from imgaug2.mlx import morphology as mlx_morph

        # Create image with white square with small hole
        image = np.ones((32, 32, 1), dtype=np.uint8) * 255
        image[15, 15] = 0  # Small hole

        result = mlx_morph.closing(image, ksize=3)

        assert result.shape == image.shape
        # Closing should fill the small hole
        assert result[15, 15, 0] == 255

    def test_morphological_gradient_detects_edges(self):
        from imgaug2.mlx import morphology as mlx_morph

        # Create image with white square
        image = np.zeros((32, 32, 1), dtype=np.uint8)
        image[10:22, 10:22] = 255

        result = mlx_morph.morphological_gradient(image, ksize=3)

        assert result.shape == image.shape
        # Gradient should be zero inside and outside the square
        assert result[15, 15, 0] == 0  # Inside
        assert result[5, 5, 0] == 0  # Outside
        # Gradient should be non-zero at edges
        assert result[10, 15, 0] > 0  # Edge


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsJpegCompression(unittest.TestCase):
    def test_jpeg_compression_produces_artifacts(self):
        from imgaug2.mlx import compression as mlx_comp

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result = mlx_comp.jpeg_compression(image, quality=10)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Low quality should produce different image
        assert not np.array_equal(result, image)

    def test_jpeg_compression_quality_affects_output(self):
        from imgaug2.mlx import compression as mlx_comp

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result_low = mlx_comp.jpeg_compression(image, quality=10)
        result_high = mlx_comp.jpeg_compression(image, quality=95)

        assert result_low.shape == image.shape
        assert result_high.shape == image.shape

        # Lower quality should have more artifacts (different from higher quality)
        diff_low = np.abs(result_low.astype(np.float32) - image.astype(np.float32))
        diff_high = np.abs(result_high.astype(np.float32) - image.astype(np.float32))
        # Lower quality typically means more difference from original
        assert diff_low.mean() >= diff_high.mean()

    def test_jpeg_compression_batch(self):
        from imgaug2.mlx import compression as mlx_comp

        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)

        result = mlx_comp.jpeg_compression(batch, quality=50)

        assert result.shape == batch.shape
        assert result.dtype == np.uint8

    def test_jpeg_compression_grayscale(self):
        from imgaug2.mlx import compression as mlx_comp

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 1), dtype=np.uint8)

        result = mlx_comp.jpeg_compression(image, quality=50)

        assert result.shape == image.shape
        assert result.dtype == np.uint8


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsSepia(unittest.TestCase):
    def test_sepia_changes_colors(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_color.sepia(image, strength=1.0)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Sepia should change the image
        assert not np.array_equal(result, image)

    def test_sepia_strength_zero_is_identity(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_color.sepia(image, strength=0.0)

        assert result.shape == image.shape
        # Strength 0 should return original (with possible minor rounding)
        diff = np.abs(result.astype(np.float32) - image.astype(np.float32))
        assert diff.max() <= 1

    def test_sepia_warm_tone(self):
        from imgaug2.mlx import color as mlx_color

        # Create a neutral gray image
        image = np.full((32, 32, 3), 128, dtype=np.uint8)

        result = mlx_color.sepia(image, strength=1.0)

        # Sepia should produce warmer (more red/yellow) tones
        # For gray input, R >= G >= B in sepia output
        avg_r = result[:, :, 0].mean()
        avg_g = result[:, :, 1].mean()
        avg_b = result[:, :, 2].mean()
        assert avg_r >= avg_g >= avg_b


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsFancyPCA(unittest.TestCase):
    def test_fancy_pca_changes_image(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_color.fancy_pca(image, alpha_std=0.1, seed=42)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should modify the image
        assert not np.array_equal(result, image)

    def test_fancy_pca_reproducible(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result1 = mlx_color.fancy_pca(image, alpha_std=0.1, seed=42)
        result2 = mlx_color.fancy_pca(image, alpha_std=0.1, seed=42)

        np.testing.assert_array_equal(result1, result2)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsIsoNoise(unittest.TestCase):
    def test_iso_noise_adds_noise(self):
        from imgaug2.mlx import noise as mlx_noise

        image = np.full((32, 32, 3), 128, dtype=np.uint8)

        result = mlx_noise.iso_noise(image, color_shift=0.05, intensity=0.5, seed=42)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should add noise
        assert not np.array_equal(result, image)
        # Check that result is still in valid range
        assert result.min() >= 0 and result.max() <= 255

    def test_iso_noise_batch(self):
        from imgaug2.mlx import noise as mlx_noise

        batch = np.full((2, 32, 32, 3), 128, dtype=np.uint8)

        result = mlx_noise.iso_noise(batch, intensity=0.5, seed=42)

        assert result.shape == batch.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsShotNoise(unittest.TestCase):
    def test_shot_noise_signal_dependent(self):
        from imgaug2.mlx import noise as mlx_noise

        # Create image with bright and dark regions
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:32, :, :] = 200  # Bright half
        image[32:, :, :] = 50  # Dark half

        result = mlx_noise.shot_noise(image, scale=1.0, seed=42)

        assert result.shape == image.shape
        # Result should be different from input
        assert not np.array_equal(result, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsOpticalDistortion(unittest.TestCase):
    def test_optical_distortion_barrel(self):
        from imgaug2.mlx import geometry as mlx_geom

        # Create checkerboard pattern
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[::8, :, :] = 255
        image[:, ::8, :] = 255

        result = mlx_geom.optical_distortion(image, k=0.5)

        assert result.shape == image.shape
        # Should be different due to distortion
        assert not np.array_equal(result, image)

    def test_optical_distortion_pincushion(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result = mlx_geom.optical_distortion(image, k=-0.5)

        assert result.shape == image.shape
        assert not np.array_equal(result, image)

    def test_optical_distortion_zero_is_identity(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result = mlx_geom.optical_distortion(image, k=0.0)

        # k=0 should be identity
        np.testing.assert_array_equal(result, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsGridDistortion(unittest.TestCase):
    def test_grid_distortion_distorts_image(self):
        from imgaug2.mlx import geometry as mlx_geom

        # Create grid pattern
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[::8, :, :] = 255
        image[:, ::8, :] = 255

        result = mlx_geom.grid_distortion(image, num_steps=5, distort_limit=0.3, seed=42)

        assert result.shape == image.shape
        # Should be different due to distortion
        assert not np.array_equal(result, image)

    def test_grid_distortion_zero_limit_is_identity(self):
        from imgaug2.mlx import geometry as mlx_geom

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        result = mlx_geom.grid_distortion(image, distort_limit=0.0, seed=42)

        # Zero distortion should be identity
        np.testing.assert_array_equal(result, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsGlassBlur(unittest.TestCase):
    def test_glass_blur_distorts_image(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_blur.glass_blur(image, sigma=0.7, max_delta=4, iterations=2, seed=42)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should modify the image
        assert not np.array_equal(result, image)

    def test_glass_blur_preserves_mlx(self):
        from imgaug2.mlx import blur as mlx_blur

        image = mx.ones((32, 32, 3), dtype=mx.float32) * 128

        result = mlx_blur.glass_blur(image, sigma=0.5, max_delta=2, iterations=1, seed=42)

        assert isinstance(result, mx.array)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsZoomBlur(unittest.TestCase):
    def test_zoom_blur_creates_radial_effect(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_blur.zoom_blur(image, max_factor=0.1, step_factor=0.02)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should blur the image
        assert not np.array_equal(result, image)

    def test_zoom_blur_batch(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        batch = rng.integers(0, 256, size=(2, 32, 32, 3), dtype=np.uint8)

        result = mlx_blur.zoom_blur(batch, max_factor=0.05, step_factor=0.02)

        assert result.shape == batch.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsDefocusBlur(unittest.TestCase):
    def test_defocus_blur_creates_disk_blur(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_blur.defocus_blur(image, radius=3)

        assert result.shape == image.shape
        assert result.dtype == np.uint8
        # Should blur the image
        assert not np.array_equal(result, image)

    def test_defocus_blur_radius_0_returns_original(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        result = mlx_blur.defocus_blur(image, radius=0)

        # Radius 0 should return original
        np.testing.assert_array_equal(result, image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsAverageBlur(unittest.TestCase):
    def test_matches_cv2_uint8(self):
        from imgaug2.mlx import blur as mlx_blur

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 40, 3), dtype=np.uint8)

        observed = mlx_blur.average_blur(image, ksize=(3, 5))
        expected = cv2.blur(
            image, (5, 3), borderType=cv2.BORDER_REFLECT_101  # cv2 uses (w, h)
        )

        diff = np.abs(observed.astype(np.int16) - expected.astype(np.int16))
        assert observed.dtype == np.uint8
        assert diff.max() <= 2


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsNoiseExtras(unittest.TestCase):
    def test_additive_gaussian_noise_is_deterministic(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out1 = mlx_noise.additive_gaussian_noise(image, scale=5.0, seed=123)
        out2 = mlx_noise.additive_gaussian_noise(image, scale=5.0, seed=123)

        np.testing.assert_array_equal(out1, out2)
        assert out1.dtype == np.uint8

    def test_coarse_dropout_is_deterministic(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out1 = mlx_noise.coarse_dropout(image, p=0.3, size_px=4, seed=7)
        out2 = mlx_noise.coarse_dropout(image, p=0.3, size_px=4, seed=7)

        np.testing.assert_array_equal(out1, out2)
        assert out1.shape == image.shape

    def test_salt_and_pepper_shape(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)

        out = mlx_noise.salt_and_pepper(image, p=0.2, seed=9)
        assert out.shape == image.shape

    def test_spatter_shape(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)

        out = mlx_noise.spatter(image, intensity=0.3, seed=11)
        assert out.shape == image.shape

    def test_pixel_shuffle_shape(self):
        from imgaug2.mlx import noise as mlx_noise

        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        out = mlx_noise.pixel_shuffle(image, block_size=4, seed=13)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsColorExtras(unittest.TestCase):
    def test_planckian_jitter_shape(self):
        from imgaug2.mlx import color as mlx_color

        rng = np.random.default_rng(5)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        out = mlx_color.planckian_jitter(image, seed=42)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsPointwiseExtras(unittest.TestCase):
    def test_linear_contrast_shape(self):
        from imgaug2.mlx import pointwise as mlx_pointwise

        rng = np.random.default_rng(6)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        out = mlx_pointwise.linear_contrast(image, factor=1.1)
        assert out.shape == image.shape

    def test_sigmoid_and_log_contrast_shape(self):
        from imgaug2.mlx import pointwise as mlx_pointwise

        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)

        out_sigmoid = mlx_pointwise.sigmoid_contrast(image, gain=5.0, cutoff=0.5)
        out_log = mlx_pointwise.log_contrast(image, gain=1.2)

        assert out_sigmoid.shape == image.shape
        assert out_log.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsGeometryExtras(unittest.TestCase):
    def test_resize_shape(self):
        from imgaug2.mlx import geometry as mlx_geom

        image = mx.ones((10, 12, 3), dtype=mx.float32)
        out = mlx_geom.resize(image, (20, 24), order=1)
        assert out.shape == (20, 24, 3)

    def test_grid_sample_shape(self):
        from imgaug2.mlx import geometry as mlx_geom

        image = mx.ones((1, 8, 8, 3), dtype=mx.float32)
        coords = mx.ones((1, 8, 8, 2), dtype=mx.float32)
        out = mlx_geom.grid_sample(image, coords, mode="bilinear", padding_mode="border")
        assert out.shape == image.shape

    def test_chromatic_aberration_shape(self):
        from imgaug2.mlx import geometry as mlx_geom

        image = mx.ones((16, 16, 3), dtype=mx.float32)
        out = mlx_geom.chromatic_aberration(
            image,
            primary_distortion=0.01,
            secondary_distortion=-0.01,
        )
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsCoreHelpers(unittest.TestCase):
    def test_core_helpers(self):
        from imgaug2.mlx import ensure_float32, is_mlx_array, require, restore_dtype

        require()
        image = mx.ones((4, 4, 3), dtype=mx.float16)
        assert is_mlx_array(image)
        image_f32 = ensure_float32(image)
        assert image_f32.dtype == mx.float32

        restored = restore_dtype(image_f32, np.dtype("uint8"), is_input_mlx=False)
        assert isinstance(restored, np.ndarray)
        assert restored.dtype == np.uint8


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsBlend(unittest.TestCase):
    def test_blend_alpha_matches_utils(self):
        from imgaug2.augmenters import _blend_utils
        from imgaug2.mlx import blend as mlx_blend

        rng = np.random.default_rng(10)
        image_fg = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        image_bg = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        alpha = 0.35

        observed = mlx_blend.blend_alpha(image_fg, image_bg, alpha)
        expected = _blend_utils.blend_alpha(image_fg, image_bg, alpha)

        np.testing.assert_array_equal(observed, expected)

    def test_blend_alpha_preserves_mlx(self):
        from imgaug2.mlx import blend as mlx_blend

        image_fg = mx.array(np.ones((8, 8, 3), dtype=np.uint8))
        image_bg = mx.array(np.zeros((8, 8, 3), dtype=np.uint8))

        out = mlx_blend.blend_alpha(image_fg, image_bg, 0.5)
        assert isinstance(out, mx.array)
        assert out.shape == image_fg.shape

    def test_blend_alpha_supports_batch(self):
        from imgaug2.mlx import blend as mlx_blend

        image_fg = mx.ones((4, 16, 16, 3), dtype=mx.float32)
        image_bg = mx.zeros((4, 16, 16, 3), dtype=mx.float32)

        out = mlx_blend.blend_alpha(image_fg, image_bg, 0.5)
        assert isinstance(out, mx.array)
        assert out.shape == image_fg.shape
        np.testing.assert_allclose(np.array(out), 0.5)

    def test_blend_alpha_supports_batch_grayscale(self):
        from imgaug2.mlx import blend as mlx_blend

        image_fg = mx.ones((3, 12, 12), dtype=mx.float32)
        image_bg = mx.zeros((3, 12, 12), dtype=mx.float32)

        out = mlx_blend.blend_alpha(image_fg, image_bg, 0.5)
        assert isinstance(out, mx.array)
        assert out.shape == image_fg.shape
        np.testing.assert_allclose(np.array(out), 0.5)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsEdges(unittest.TestCase):
    def test_canny_thresholds_reduce_edges(self):
        from imgaug2.mlx import edges as mlx_edges

        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = 255

        low = mlx_edges.canny(image, 10, 20, sobel_kernel_size=3)
        high = mlx_edges.canny(image, 100, 200, sobel_kernel_size=3)

        assert low.shape == (32, 32)
        assert high.shape == (32, 32)
        assert low.dtype == np.bool_
        assert high.dtype == np.bool_
        assert np.any(low)
        assert np.sum(high) <= np.sum(low)

    def test_canny_returns_mlx(self):
        from imgaug2.mlx import edges as mlx_edges

        image = mx.array(np.ones((16, 16, 3), dtype=np.uint8))

        out = mlx_edges.canny(image, 10, 20)
        assert isinstance(out, mx.array)
        assert out.shape == (16, 16)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsConvolutional(unittest.TestCase):
    def test_convolve_matches_cpu(self):
        from imgaug2.augmenters import convolutional as conv_cpu
        from imgaug2.mlx import convolutional as mlx_conv

        rng = np.random.default_rng(12)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        kernel /= kernel.sum()

        observed = mlx_conv.convolve(image, kernel)
        expected = conv_cpu.convolve(image, kernel)

        np.testing.assert_array_equal(observed, expected)

    def test_convolve_matches_cpu_per_channel_kernels_and_even_kernel(self):
        from imgaug2.augmenters import convolutional as conv_cpu
        from imgaug2.mlx import convolutional as mlx_conv

        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, size=(31, 27, 3), dtype=np.uint8)

        # Mix: a 2x2 kernel (even), a skipped channel, and another 2x2 kernel.
        kernels = [
            np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32),
            None,
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ]

        observed = mlx_conv.convolve(image, kernels)
        expected = conv_cpu.convolve(image, kernels)

        np.testing.assert_array_equal(observed, expected)

    def test_convolve_matches_cpu_grayscale_bool(self):
        from imgaug2.augmenters import convolutional as conv_cpu
        from imgaug2.mlx import convolutional as mlx_conv

        rng = np.random.default_rng(7)
        image = rng.random((25, 19)) > 0.7
        image = image.astype(np.bool_)

        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        observed = mlx_conv.convolve(image, kernel)
        expected = conv_cpu.convolve(image, kernel)

        np.testing.assert_array_equal(observed, expected)

    def test_convolve_returns_mlx(self):
        from imgaug2.mlx import convolutional as mlx_conv

        image = mx.array(np.ones((8, 8, 3), dtype=np.uint8))
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

        out = mlx_conv.convolve(image, kernel)
        assert isinstance(out, mx.array)
        assert out.shape == image.shape

    def test_convolve_accepts_batched_nhwc_mlx(self):
        from imgaug2.augmenters import convolutional as conv_cpu
        from imgaug2.mlx import convolutional as mlx_conv

        rng = np.random.default_rng(11)
        images = rng.integers(0, 256, size=(4, 32, 33, 3), dtype=np.uint8)
        kernel = np.array(
            [[0.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        kernel /= kernel.sum()

        out = mlx_conv.convolve(mx.array(images), kernel)
        assert isinstance(out, mx.array)
        assert out.shape == images.shape

        expected = np.stack([conv_cpu.convolve(images[i], kernel) for i in range(images.shape[0])], axis=0)
        np.testing.assert_array_equal(np.array(out), expected)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsArtistic(unittest.TestCase):
    def test_stylize_cartoon_matches_cpu(self):
        from imgaug2.augmenters import artistic as cpu_artistic
        from imgaug2.mlx import artistic as mlx_artistic

        rng = np.random.default_rng(13)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        observed = mlx_artistic.stylize_cartoon(
            image,
            blur_ksize=3,
            segmentation_size=1.0,
            saturation=1.2,
            edge_prevalence=1.0,
            suppress_edges=False,
        )
        expected = cpu_artistic.stylize_cartoon(
            image,
            blur_ksize=3,
            segmentation_size=1.0,
            saturation=1.2,
            edge_prevalence=1.0,
            suppress_edges=False,
        )

        np.testing.assert_array_equal(observed, expected)

    def test_stylize_cartoon_preserves_mlx(self):
        from imgaug2.mlx import artistic as mlx_artistic

        image = mx.array(np.zeros((32, 32, 3), dtype=np.uint8))

        out = mlx_artistic.stylize_cartoon(image, blur_ksize=3, suppress_edges=False)
        assert isinstance(out, mx.array)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsPillike(unittest.TestCase):
    def test_autocontrast_matches_cpu(self):
        from imgaug2.augmenters import pillike as cpu_pillike
        from imgaug2.mlx import pillike as mlx_pillike

        rng = np.random.default_rng(14)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

        observed = mlx_pillike.autocontrast(image, cutoff=2)
        expected = cpu_pillike.autocontrast(image, cutoff=2)

        np.testing.assert_array_equal(observed, expected)

    def test_equalize_preserves_mlx(self):
        from imgaug2.mlx import pillike as mlx_pillike

        rng = np.random.default_rng(15)
        image = mx.array(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8))
        mask = mx.array(rng.integers(0, 2, size=(32, 32), dtype=np.uint8) * 255)

        out = mlx_pillike.equalize(image, mask=mask, allow_cpu_fallback=True)
        assert isinstance(out, mx.array)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsSegmentation(unittest.TestCase):
    def test_segment_voronoi_matches_cpu(self):
        from imgaug2.augmenters import segmentation as cpu_seg
        from imgaug2.mlx import segmentation as mlx_seg

        rng = np.random.default_rng(16)
        image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        coords = np.array([[8.0, 8.0], [24.0, 24.0]], dtype=np.float32)

        observed = mlx_seg.segment_voronoi(image, coords)
        expected = cpu_seg.segment_voronoi(image, coords)

        np.testing.assert_array_equal(observed, expected)

    def test_replace_segments_preserves_mlx(self):
        from imgaug2.mlx import segmentation as mlx_seg

        image = mx.array(np.zeros((16, 16, 3), dtype=np.uint8))
        segments = mx.array(np.zeros((16, 16), dtype=np.int32))
        replace_flags = mx.array(np.array([True], dtype=bool))

        out = mlx_seg.replace_segments_(image, segments, replace_flags)
        assert isinstance(out, mx.array)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxOpsImgcorruptlike(unittest.TestCase):
    def test_apply_gaussian_noise_deterministic(self):
        from imgaug2.mlx import imgcorruptlike as mlx_corrupt

        rng = np.random.default_rng(17)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)

        observed_a = mlx_corrupt.apply_gaussian_noise(image, severity=2, seed=123)
        observed_b = mlx_corrupt.apply_gaussian_noise(image, severity=2, seed=123)

        np.testing.assert_array_equal(observed_a, observed_b)
        assert observed_a.dtype == np.uint8
        assert observed_a.shape == image.shape

    def test_apply_gaussian_noise_preserves_mlx(self):
        from imgaug2.mlx import imgcorruptlike as mlx_corrupt

        image = mx.array(np.zeros((32, 32, 3), dtype=np.uint8))
        out = mlx_corrupt.apply_gaussian_noise(image, severity=1, seed=7)
        assert isinstance(out, mx.array)
        assert out.shape == image.shape


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxBackendPolicy(unittest.TestCase):
    def test_affine_transform_requires_fallback_flag(self):
        from imgaug2.mlx import geometry as mlx_geometry

        image = mx.ones((16, 16, 3), dtype=mx.float32)
        matrix = np.eye(3, dtype=np.float32)

        with self.assertRaises(NotImplementedError):
            mlx_geometry.affine_transform(image, matrix, order=2)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxPillikePolicy(unittest.TestCase):
    def test_pillike_requires_fallback_flag(self):
        from imgaug2.mlx import pillike as mlx_pillike

        image = mx.ones((16, 16, 3), dtype=mx.uint8)

        with self.assertRaises(NotImplementedError):
            mlx_pillike.autocontrast(image)


@unittest.skipIf(not MLX_AVAILABLE, "mlx not installed")
class TestMlxArithmeticOps(unittest.TestCase):
    def test_add_elementwise_matches_numpy(self):
        import imgaug2.augmenters.arithmetic as iaa_arith

        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(12, 10, 3), dtype=np.uint8)
        values = rng.integers(-5, 6, size=(12, 10, 1), dtype=np.int16)

        expected = iaa_arith.add_elementwise(np.copy(image), values)
        observed = iaa_arith.add_elementwise(mx.array(image), mx.array(values))

        np.testing.assert_array_equal(np.array(observed), expected)

    def test_invert_matches_numpy(self):
        import imgaug2.augmenters.arithmetic as iaa_arith

        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(8, 9, 3), dtype=np.uint8)

        expected = iaa_arith.invert_(np.copy(image))
        observed = iaa_arith.invert_(mx.array(image))

        np.testing.assert_array_equal(np.array(observed), expected)

    def test_replace_elementwise_matches_numpy(self):
        import imgaug2.augmenters.arithmetic as iaa_arith

        rng = np.random.default_rng(2)
        image = rng.integers(0, 256, size=(6, 7, 3), dtype=np.uint8)
        mask = np.zeros((6, 7, 1), dtype=np.float32)
        mask[1, 2, 0] = 1.0
        mask[4, 5, 0] = 1.0
        mask_broadcast = np.broadcast_to(mask > 0.5, image.shape)
        num_replace = int(mask_broadcast.sum())
        replacements = rng.integers(0, 256, size=(num_replace,), dtype=np.uint8)

        expected = iaa_arith.replace_elementwise_(np.copy(image), mask, replacements)
        observed = iaa_arith.replace_elementwise_(
            mx.array(image), mx.array(mask), mx.array(replacements)
        )

        np.testing.assert_array_equal(np.array(observed), expected)

    def test_cutout_constant_matches_numpy(self):
        import imgaug2.augmenters.arithmetic as iaa_arith

        rng = np.random.default_rng(3)
        image = rng.integers(0, 256, size=(9, 11, 3), dtype=np.uint8)

        expected = iaa_arith.cutout_(
            np.copy(image),
            x1=2,
            y1=3,
            x2=8,
            y2=7,
            fill_mode="constant",
            cval=(10, 20, 30),
            fill_per_channel=True,
            seed=1,
        )
        observed = iaa_arith.cutout_(
            mx.array(image),
            x1=2,
            y1=3,
            x2=8,
            y2=7,
            fill_mode="constant",
            cval=(10, 20, 30),
            fill_per_channel=True,
            seed=1,
        )

        np.testing.assert_array_equal(np.array(observed), expected)
