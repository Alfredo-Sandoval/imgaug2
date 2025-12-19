import unittest
from unittest import mock

import numpy as np

import imgaug2 as ia
from imgaug2.augmentables.utils import (
    copy_augmentables,
    deepcopy_fast,
    interpolate_point_pair,
    interpolate_points,
    interpolate_points_by_max_distance,
    normalize_imglike_shape,
    normalize_shape,
)


class Test_interpolate_point_pair(unittest.TestCase):
    def test_1_step(self):
        point_a = (0, 0)
        point_b = (1, 2)
        inter = interpolate_point_pair(point_a, point_b, 1)
        assert np.allclose(inter, np.float32([[0.5, 1.0]]))

    def test_2_steps(self):
        point_a = (0, 0)
        point_b = (1, 2)
        inter = interpolate_point_pair(point_a, point_b, 2)
        assert np.allclose(inter, np.float32([[1 * 1 / 3, 1 * 2 / 3], [2 * 1 / 3, 2 * 2 / 3]]))

    def test_0_steps(self):
        point_a = (0, 0)
        point_b = (1, 2)
        inter = interpolate_point_pair(point_a, point_b, 0)
        assert len(inter) == 0


class Test_interpolate_points(unittest.TestCase):
    def test_2_points_0_steps(self):
        points = [(0, 0), (1, 2)]

        inter = interpolate_points(points, 0)

        assert np.allclose(inter, np.float32([[0, 0], [1, 2]]))

    def test_2_points_1_step(self):
        points = [(0, 0), (1, 2)]

        inter = interpolate_points(points, 1)

        assert np.allclose(inter, np.float32([[0, 0], [0.5, 1.0], [1, 2], [0.5, 1.0]]))

    def test_2_points_1_step_not_closed(self):
        points = [(0, 0), (1, 2)]

        inter = interpolate_points(points, 1, closed=False)

        assert np.allclose(inter, np.float32([[0, 0], [0.5, 1.0], [1, 2]]))

    def test_3_points_0_steps(self):
        points = [(0, 0), (1, 2), (0.5, 3)]

        inter = interpolate_points(points, 0)

        assert np.allclose(inter, np.float32([[0, 0], [1, 2], [0.5, 3]]))

    def test_3_points_1_step(self):
        points = [(0, 0), (1, 2), (0.5, 3)]

        inter = interpolate_points(points, 1)

        assert np.allclose(
            inter, np.float32([[0, 0], [0.5, 1.0], [1, 2], [0.75, 2.5], [0.5, 3], [0.25, 1.5]])
        )

    def test_3_points_1_step_not_closed(self):
        points = [(0, 0), (1, 2), (0.5, 3)]

        inter = interpolate_points(points, 1, closed=False)

        assert np.allclose(inter, np.float32([[0, 0], [0.5, 1.0], [1, 2], [0.75, 2.5], [0.5, 3]]))

    def test_0_points_1_step(self):
        points = []

        inter = interpolate_points(points, 1)

        assert len(inter) == 0

    def test_1_point_0_steps(self):
        points = [(0, 0)]

        inter = interpolate_points(points, 0)

        assert np.allclose(inter, np.float32([[0, 0]]))

    def test_1_point_1_step(self):
        points = [(0, 0)]

        inter = interpolate_points(points, 1)

        assert np.allclose(inter, np.float32([[0, 0]]))


class Test_interpolate_points_by_max_distance(unittest.TestCase):
    def test_2_points_dist_10000(self):
        points = [(0, 0), (0, 2)]

        inter = interpolate_points_by_max_distance(points, 10000)

        assert np.allclose(inter, points)

    def test_2_points_dist_1(self):
        points = [(0, 0), (0, 2)]

        inter = interpolate_points_by_max_distance(points, 1.0)

        assert np.allclose(inter, np.float32([[0, 0], [0, 1.0], [0, 2], [0, 1.0]]))

    def test_2_points_dist_1_not_closed(self):
        points = [(0, 0), (0, 2)]

        inter = interpolate_points_by_max_distance(points, 1.0, closed=False)

        assert np.allclose(inter, np.float32([[0, 0], [0, 1.0], [0, 2]]))

    def test_3_points_dist_1(self):
        points = [(0, 0), (0, 2), (2, 0)]

        inter = interpolate_points_by_max_distance(points, 1.0)
        assert np.allclose(
            inter, np.float32([[0, 0], [0, 1.0], [0, 2], [1.0, 1.0], [2, 0], [1.0, 0]])
        )

    def test_3_points_dist_1_not_closed(self):
        points = [(0, 0), (0, 2), (2, 0)]

        inter = interpolate_points_by_max_distance(points, 1.0, closed=False)

        assert np.allclose(inter, np.float32([[0, 0], [0, 1.0], [0, 2], [1.0, 1.0], [2, 0]]))

    def test_0_points_dist_1(self):
        points = []

        inter = interpolate_points_by_max_distance(points, 1.0)

        assert len(inter) == 0

    def test_1_point_dist_1(self):
        points = [(0, 0)]

        inter = interpolate_points_by_max_distance(points, 1.0)

        assert np.allclose(inter, np.float32([[0, 0]]))


class Test_normalize_shape(unittest.TestCase):
    def test_shape_tuple(self):
        shape_out = normalize_shape((1, 2))
        assert shape_out == (1, 2)

    def test_shape_tuple_3d(self):
        shape_out = normalize_shape((1, 2, 3))
        assert shape_out == (1, 2, 3)

    def test_array_1d(self):
        arr = np.zeros((5,), dtype=np.uint8)
        shape_out = normalize_shape(arr)
        assert shape_out == (5,)

    def test_array_2d(self):
        arr = np.zeros((1, 2), dtype=np.uint8)
        shape_out = normalize_shape(arr)
        assert shape_out == (1, 2)

    def test_array_3d(self):
        arr = np.zeros((1, 2, 3), dtype=np.uint8)
        shape_out = normalize_shape(arr)
        assert shape_out == (1, 2, 3)


class Test_normalize_imglike_shape(unittest.TestCase):
    def test_shape_tuple(self):
        shape_out = normalize_imglike_shape((1, 2))
        assert shape_out == (1, 2)

    def test_shape_tuple_3d(self):
        shape_out = normalize_imglike_shape((1, 2, 3))
        assert shape_out == (1, 2, 3)

    def test_array_1d_fails(self):
        arr = np.zeros((5,), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            _ = normalize_imglike_shape(arr)

    def test_array_2d(self):
        arr = np.zeros((1, 2), dtype=np.uint8)
        shape_out = normalize_imglike_shape(arr)
        assert shape_out == (1, 2)

    def test_array_3d(self):
        arr = np.zeros((1, 2, 3), dtype=np.uint8)
        shape_out = normalize_imglike_shape(arr)
        assert shape_out == (1, 2, 3)

    def test_array_4d_fails(self):
        arr = np.zeros((1, 2, 3, 4), dtype=np.uint8)
        with self.assertRaises(AssertionError):
            _ = normalize_imglike_shape(arr)


class Test_copy_augmentables(unittest.TestCase):
    def test_numpy_array_is_copied(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)

        result = copy_augmentables(arr)

        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
        assert result is not arr

    def test_numpy_array_modification_does_not_affect_original(self):
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        original_value = arr[0, 0]

        result = copy_augmentables(arr)
        result[0, 0] = 255

        assert arr[0, 0] == original_value
        assert result[0, 0] == 255

    def test_list_of_numpy_arrays(self):
        arr1 = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        arr2 = np.array([[5, 6], [7, 8]], dtype=np.uint8)
        augmentables = [arr1, arr2]

        result = copy_augmentables(augmentables)

        assert isinstance(result, list)
        assert len(result) == 2
        assert np.array_equal(result[0], arr1)
        assert np.array_equal(result[1], arr2)
        assert result[0] is not arr1
        assert result[1] is not arr2

    def test_list_of_keypoints_on_image(self):
        kpsoi1 = ia.KeypointsOnImage(
            [ia.Keypoint(x=1, y=2), ia.Keypoint(x=3, y=4)],
            shape=(10, 10, 3),
        )
        kpsoi2 = ia.KeypointsOnImage(
            [ia.Keypoint(x=5, y=6)],
            shape=(20, 20, 3),
        )
        augmentables = [kpsoi1, kpsoi2]

        result = copy_augmentables(augmentables)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] is not kpsoi1
        assert result[1] is not kpsoi2
        assert len(result[0].keypoints) == 2
        assert len(result[1].keypoints) == 1
        assert result[0].keypoints[0].x == 1
        assert result[0].keypoints[0].y == 2

    def test_list_of_bounding_boxes_on_image(self):
        bbsoi = ia.BoundingBoxesOnImage(
            [ia.BoundingBox(x1=0, y1=0, x2=10, y2=10)],
            shape=(50, 50, 3),
        )
        augmentables = [bbsoi]

        result = copy_augmentables(augmentables)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not bbsoi
        assert len(result[0].bounding_boxes) == 1
        assert result[0].bounding_boxes[0].x1 == 0
        assert result[0].bounding_boxes[0].x2 == 10

    def test_list_of_polygons_on_image(self):
        psoi = ia.PolygonsOnImage(
            [ia.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
            shape=(50, 50, 3),
        )
        augmentables = [psoi]

        result = copy_augmentables(augmentables)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] is not psoi
        assert len(result[0].polygons) == 1

    def test_mixed_list(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        kpsoi = ia.KeypointsOnImage(
            [ia.Keypoint(x=1, y=2)],
            shape=(10, 10, 3),
        )
        augmentables = [arr, kpsoi]

        result = copy_augmentables(augmentables)

        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], ia.KeypointsOnImage)
        assert result[0] is not arr
        assert result[1] is not kpsoi


class Test_deepcopy_fast(unittest.TestCase):
    def test_none_returns_none(self):
        result = deepcopy_fast(None)
        assert result is None

    def test_integer_returns_same_value(self):
        result = deepcopy_fast(42)
        assert result == 42

    def test_float_returns_same_value(self):
        result = deepcopy_fast(3.14)
        assert result == 3.14

    def test_string_returns_same_value(self):
        result = deepcopy_fast("hello")
        assert result == "hello"

    def test_list_is_deepcopied(self):
        original = [1, 2, [3, 4]]
        result = deepcopy_fast(original)
        assert result == original
        assert result is not original
        assert result[2] is not original[2]

    def test_tuple_is_deepcopied(self):
        original = (1, 2, [3, 4])
        result = deepcopy_fast(original)
        assert result == original
        # tuples are immutable so outer can be same, but nested list should differ
        assert result[2] is not original[2]

    def test_numpy_array_is_copied(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result = deepcopy_fast(arr)
        assert np.array_equal(result, arr)
        assert result is not arr

    def test_numpy_array_modification_does_not_affect_original(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        original_value = arr[0, 0]
        result = deepcopy_fast(arr)
        result[0, 0] = 255
        assert arr[0, 0] == original_value
        assert result[0, 0] == 255

    def test_object_with_deepcopy_method(self):
        kpsoi = ia.KeypointsOnImage(
            [ia.Keypoint(x=1, y=2)],
            shape=(10, 10, 3),
        )
        result = deepcopy_fast(kpsoi)
        assert result is not kpsoi
        assert len(result.keypoints) == 1
        assert result.keypoints[0].x == 1
        assert result.keypoints[0].y == 2

    def test_nested_list_is_deepcopied_recursively(self):
        nested = [[1, 2], [3, [4, 5]]]
        result = deepcopy_fast(nested)
        assert result == nested
        assert result is not nested
        assert result[0] is not nested[0]
        assert result[1] is not nested[1]
        assert result[1][1] is not nested[1][1]

    def test_nested_tuple_with_list(self):
        nested = ((1, 2), [3, 4])
        result = deepcopy_fast(nested)
        assert result[1] is not nested[1]
        result[1][0] = 999
        assert nested[1][0] == 3

    def test_object_without_deepcopy_uses_copy_deepcopy(self):
        class CustomClass:
            def __init__(self, value):
                self.value = value

        obj = CustomClass(42)
        result = deepcopy_fast(obj)
        assert result is not obj
        assert result.value == 42
