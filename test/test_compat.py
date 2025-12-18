import unittest

import numpy as np

from imgaug2 import compat as A


class TestCompatCompose(unittest.TestCase):
    def test_horizontal_flip_pascal_voc_bboxes_and_keypoints(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        bboxes = [(10, 10, 20, 20)]
        keypoints = [(5.5, 6.25)]

        transform = A.Compose(
            [A.HorizontalFlip(p=1.0)],
            bbox_params=A.BboxParams(format="pascal_voc"),
            keypoint_params=A.KeypointParams(),
        )
        out = transform(image=image, bboxes=bboxes, keypoints=keypoints)

        assert out["image"].shape == image.shape
        assert out["bboxes"] == [(44.0, 10.0, 54.0, 20.0)]
        assert out["keypoints"] == [(58.5, 6.25)]

    def test_horizontal_flip_coco_with_label_fields(self):
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        # COCO format: (x_min, y_min, width, height)
        bboxes = [(10, 10, 10, 10)]
        category_ids = [7]

        transform = A.Compose(
            [A.HorizontalFlip(p=1.0)],
            bbox_params=A.BboxParams(format="coco", label_fields=("category_ids",)),
        )
        out = transform(image=image, bboxes=bboxes, category_ids=category_ids)

        assert out["bboxes"] == [(44.0, 10.0, 10.0, 10.0)]
        assert out["category_ids"] == [7]

    def test_horizontal_flip_yolo_is_symmetric(self):
        image = np.zeros((50, 100, 3), dtype=np.uint8)

        # YOLO format: (x_center, y_center, width, height), all normalized
        bboxes = [(0.5, 0.5, 0.2, 0.4)]

        transform = A.Compose([A.HorizontalFlip(p=1.0)], bbox_params=A.BboxParams(format="yolo"))
        out = transform(image=image, bboxes=bboxes)

        # Centered box should stay centered after horizontal flip.
        (xc, yc, bw, bh) = out["bboxes"][0]
        assert abs(xc - 0.5) < 1e-6
        assert abs(yc - 0.5) < 1e-6
        assert abs(bw - 0.2) < 1e-6
        assert abs(bh - 0.4) < 1e-6
