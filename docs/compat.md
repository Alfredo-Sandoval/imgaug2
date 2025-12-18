# Compat API (Dictionary-Style)

imgaug2’s native API is the classic `imgaug` style (augmenter objects + `Sequential`, `Sometimes`, etc.).

If you prefer the **Dictionary-style developer experience** (dict I/O, single call, `p=` on everything),
use the optional compat layer:

```python
from imgaug2 import compat as A

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(p=0.5),
    ],
    bbox_params=A.BboxParams(format="coco"),
    keypoint_params=A.KeypointParams(remove_invisible=True),
)

out = transform(image=image, bboxes=bboxes, keypoints=keypoints)
image_aug = out["image"]
bboxes_aug = out["bboxes"]
keypoints_aug = out["keypoints"]
```

## Bounding Boxes

`BboxParams` supports common formats:

- `pascal_voc`: `(x_min, y_min, x_max, y_max)` (absolute pixels)
- `coco`: `(x_min, y_min, width, height)` (absolute pixels)
- `yolo`: `(x_center, y_center, width, height)` (normalized 0..1)
- `xyxy_norm`: `(x_min, y_min, x_max, y_max)` (normalized 0..1)

### Labels (`label_fields`)

To keep labels in sync with boxes (like other dict-based libraries), store labels in separate lists and declare them
via `label_fields`:

```python
from imgaug2 import compat as A

transform = A.Compose(
    [A.HorizontalFlip(p=1.0)],
    bbox_params=A.BboxParams(format="coco", label_fields=("category_ids",)),
)

out = transform(image=image, bboxes=bboxes, category_ids=category_ids)
```

## Keypoints

Keypoints are **float** `(x, y)` coordinates. If `remove_invisible=True`, keypoints outside the image
are dropped (and any extra per-keypoint values are kept aligned).
