import imgaug2.api as ia


def test_api_exports_modules():
    assert hasattr(ia, "augmenters")
    assert hasattr(ia, "augmentables")
    assert hasattr(ia, "data")
    assert hasattr(ia, "parameters")
    assert hasattr(ia, "random")


def test_api_exports_core_types_and_helpers():
    assert ia.RNG is not None
    assert callable(ia.seed)
    assert hasattr(ia, "BoundingBoxesOnImage")
    assert hasattr(ia, "KeypointsOnImage")
    assert hasattr(ia, "SegmentationMapsOnImage")
