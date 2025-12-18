# imgaug2.augmentables Module

Classes for representing annotations that can be augmented alongside images.

## Bounding Boxes

### BoundingBox

Represents a single bounding box.

```python
from imgaug2.augmentables.bbs import BoundingBox

bb = BoundingBox(x1=10, y1=20, x2=100, y2=120, label="object")

# Properties
bb.x1, bb.y1, bb.x2, bb.y2  # Coordinates
bb.width, bb.height          # Dimensions
bb.area                      # Area
bb.center_x, bb.center_y     # Center point
bb.label                     # Optional label

# Methods
bb.project(from_shape, to_shape)  # Project to different image size
bb.extend(all_sides=10)           # Expand bounding box
bb.shift(x=5, y=10)               # Move bounding box
bb.iou(other_bb)                  # Intersection over Union
bb.is_out_of_image(image)         # Check if outside image
bb.clip_out_of_image(image)       # Clip to image bounds
bb.draw_on_image(image)           # Draw on image
```

### BoundingBoxesOnImage

Collection of bounding boxes for an image.

```python
from imgaug2.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

bbs = BoundingBoxesOnImage([
    BoundingBox(x1=10, y1=20, x2=100, y2=120),
    BoundingBox(x1=50, y1=60, x2=150, y2=160)
], shape=(256, 256, 3))

# Methods
bbs.on(image)                    # Project to image shape
bbs.remove_out_of_image_()       # Remove boxes outside image
bbs.clip_out_of_image_()         # Clip boxes to image
bbs.draw_on_image(image)         # Draw all boxes

# Conversion
bbs.to_xyxy_array()              # To numpy array (N, 4)
BoundingBoxesOnImage.from_xyxy_array(arr, shape)  # From array
```

## Keypoints

### Keypoint

Represents a single keypoint.

```python
from imgaug2.augmentables.kps import Keypoint

kp = Keypoint(x=50, y=100)

# Properties
kp.x, kp.y           # Coordinates (float)
kp.x_int, kp.y_int   # Integer coordinates

# Methods
kp.project(from_shape, to_shape)
kp.shift(x=5, y=10)
kp.is_out_of_image(image)
```

### KeypointsOnImage

Collection of keypoints for an image.

```python
from imgaug2.augmentables.kps import Keypoint, KeypointsOnImage

kps = KeypointsOnImage([
    Keypoint(x=50, y=100),
    Keypoint(x=75, y=150)
], shape=(256, 256, 3))

# Methods
kps.on(image)
kps.remove_out_of_image_()
kps.draw_on_image(image, size=5)
```

## Polygons

### Polygon

Represents a single polygon.

```python
from imgaug2.augmentables.polys import Polygon

poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])

# Properties
poly.exterior               # List of coordinates
poly.is_valid               # Geometry validity

# Methods
poly.project(from_shape, to_shape)
poly.shift(x=5, y=10)
poly.draw_on_image(image)
```

### PolygonsOnImage

Collection of polygons for an image.

```python
from imgaug2.augmentables.polys import Polygon, PolygonsOnImage

polys = PolygonsOnImage([
    Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
], shape=(256, 256, 3))
```

## Line Strings

### LineString

Represents a connected line.

```python
from imgaug2.augmentables.lines import LineString

ls = LineString([(0, 0), (50, 50), (100, 100)])

# Methods
ls.project(from_shape, to_shape)
ls.draw_on_image(image)
```

### LineStringsOnImage

Collection of line strings.

```python
from imgaug2.augmentables.lines import LineString, LineStringsOnImage

lss = LineStringsOnImage([
    LineString([(0, 0), (100, 100)])
], shape=(256, 256, 3))
```

## Heatmaps

### HeatmapsOnImage

Float arrays representing spatial distributions.

```python
from imgaug2.augmentables.heatmaps import HeatmapsOnImage
import numpy as np

# Create heatmap (values in [0, 1])
arr = np.random.rand(256, 256).astype(np.float32)
heatmap = HeatmapsOnImage(arr, shape=(256, 256, 3))

# Properties
heatmap.shape              # Associated image shape
heatmap.min_value          # Minimum value
heatmap.max_value          # Maximum value

# Methods
heatmap.get_arr()          # Get underlying array
heatmap.resize((H, W))     # Resize heatmap
heatmap.draw_on_image(image, alpha=0.5)
```

## Segmentation Maps

### SegmentationMapsOnImage

Integer class labels for each pixel.

```python
from imgaug2.augmentables.segmaps import SegmentationMapsOnImage
import numpy as np

# Create segmentation map (integer class labels)
arr = np.zeros((256, 256), dtype=np.int32)
arr[50:200, 50:200] = 1  # Class 1 region

segmap = SegmentationMapsOnImage(arr, shape=(256, 256, 3))

# Methods
segmap.get_arr()           # Get underlying array
segmap.resize((H, W))      # Resize (nearest neighbor)
segmap.draw_on_image(image, alpha=0.5)
```
