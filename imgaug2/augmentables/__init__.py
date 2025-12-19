"""Combination of all augmentable classes and related functions."""

from __future__ import annotations

from . import (
    base,
    batches,
    bbs,
    heatmaps,
    kps,
    lines,
    normalization,
    polys,
    segmaps,
    utils,
)

from .base import IAugmentable
from .batches import Batch, UnnormalizedBatch, DEFAULT
from .bbs import BoundingBox, BoundingBoxesOnImage
from .heatmaps import HeatmapsOnImage
from .kps import Keypoint, KeypointsOnImage, compute_geometric_median
from .lines import LineString, LineStringsOnImage, interpolate_points, normalize_imglike_shape
from .polys import MultiPolygon, Polygon, PolygonsOnImage, recover_psois_
from .segmaps import SegmentationMapOnImage, SegmentationMapsOnImage

__all__ = [
    # Base
    "IAugmentable",
    # Batches
    "Batch",
    "UnnormalizedBatch",
    "DEFAULT",
    # Bounding boxes
    "BoundingBox",
    "BoundingBoxesOnImage",
    # Heatmaps
    "HeatmapsOnImage",
    # Keypoints
    "Keypoint",
    "KeypointsOnImage",
    "compute_geometric_median",
    # Lines
    "LineString",
    "LineStringsOnImage",
    "interpolate_points",
    "normalize_imglike_shape",
    # Polygons
    "MultiPolygon",
    "Polygon",
    "PolygonsOnImage",
    "recover_psois_",
    # Segmentation maps
    "SegmentationMapOnImage",
    "SegmentationMapsOnImage",
    # Modules
    "base",
    "batches",
    "bbs",
    "heatmaps",
    "kps",
    "lines",
    "normalization",
    "polys",
    "segmaps",
    "utils",
]
