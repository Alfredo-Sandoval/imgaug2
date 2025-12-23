import numpy as np
import pytest

import imgaug2 as ia
from imgaug2 import augmenters as iaa
from imgaug2.mlx import is_available, to_mlx, to_numpy

try:
    import mlx.core as mx
except ImportError:
    mx = None

MLX_AVAILABLE = is_available()


@pytest.mark.skipif(not MLX_AVAILABLE, reason="MLX not available")
class TestMLXColor:
    def test_grayscale(self):
        # Create random RGB image
        image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

        # CPU augmentation
        aug = iaa.Grayscale(alpha=1.0)

        # MLX augmentation
        image_mlx = to_mlx(image[None, ...])  # (1, H, W, C)

        # Testing _augment_batch_ directly:
        from imgaug2.augmentables.batches import _BatchInAugmentation

        batch = _BatchInAugmentation(
            images=image_mlx,
            keypoints=[],
            bounding_boxes=[],
            polygons=[],
            line_strings=[],
            heatmaps=[],
            segmentation_maps=[],
        )
        res_batch = aug._augment_batch_(batch, ia.random.RNG(0), [], None)
        res_mlx = res_batch.images

        # Check result
        assert res_mlx.shape == (1, 128, 128, 3)
        # For Grayscale(1.0), R=G=B
        res_np = to_numpy(res_mlx[0])
        assert np.allclose(res_np[..., 0], res_np[..., 1], atol=1)
        assert np.allclose(res_np[..., 1], res_np[..., 2], atol=1)

    def test_multiply_hue_and_saturation(self):
        # Image: Green (H=120) -> [0, 255, 0]
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[..., 1] = 255  # Green

        image_mlx = to_mlx(image[None, ...])  # (1, H, W, C)

        # Multiply H by 1.5 -> 120 * 1.5 = 180 (Cyan) [0, 255, 255]
        aug = iaa.MultiplyHueAndSaturation(mul_hue=1.5, mul_saturation=1.0)

        from imgaug2.augmentables.batches import _BatchInAugmentation

        batch = _BatchInAugmentation(
            images=image_mlx,
            keypoints=[],
            bounding_boxes=[],
            polygons=[],
            line_strings=[],
            heatmaps=[],
            segmentation_maps=[],
        )
        res_batch = aug._augment_batch_(batch, ia.random.RNG(0), [], None)
        res_mlx = res_batch.images

        res_np = to_numpy(res_mlx[0])
        # Check center pixel
        pixel = res_np[5, 5]

        # Check if output is in 0-1 range
        if res_np.max() <= 1.0:
            scale = 1.0
        else:
            scale = 255.0

        # Expect Cyan [0, 1, 1] (Green+Blue)
        # R should be low, G and B high.
        assert pixel[0] < 0.2 * scale  # R (0)
        assert pixel[1] > 0.8 * scale  # G (1)
        assert pixel[2] > 0.8 * scale  # B (1)

    def test_add_to_hue_and_saturation(self):
        # Image: Red (H=0) -> [255, 0, 0]
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[..., 0] = 255

        image_mlx = to_mlx(image[None, ...])  # (1, H, W, C)

        # Add +120 degrees to Hue.
        # Input 85 -> 60 OpenCV units = 120 degrees.
        aug = iaa.AddToHueAndSaturation(value_hue=85)

        from imgaug2.augmentables.batches import _BatchInAugmentation

        batch = _BatchInAugmentation(
            images=image_mlx,
            keypoints=[],
            bounding_boxes=[],
            polygons=[],
            line_strings=[],
            heatmaps=[],
            segmentation_maps=[],
        )
        res_batch = aug._augment_batch_(batch, ia.random.RNG(0), [], None)
        res_mlx = res_batch.images

        res_np = to_numpy(res_mlx[0])
        pixel = res_np[5, 5]

        if res_np.max() <= 1.0:
            scale = 1.0
        else:
            scale = 255.0

        # Red(0) + 120 deg = Green (120) [0, 1, 0]
        # R=0, G=1, B=0
        assert pixel[0] < 0.2 * scale  # R
        assert pixel[1] > 0.8 * scale  # G
        assert pixel[2] < 0.2 * scale  # B
