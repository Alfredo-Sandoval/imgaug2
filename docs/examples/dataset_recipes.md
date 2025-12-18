# Dataset Recipes (Real-World Gotchas)

This page covers a few “not imgaug2’s job, but your training pipeline must handle it”
recipes. These issues show up in real datasets and are easy to miss.

## Horizontal Flip: Swap Left/Right Semantics (Keypoints)

`Fliplr` correctly flips **coordinates**, but it cannot know which keypoints are
“left” vs “right”. If your model expects a consistent semantic ordering, you
must swap indices after the flip.

Example (toy swap):

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa
import imgaug2.data as data

image = data.quokka(size=(256, 256))
kps = data.quokka_keypoints(size=(256, 256))

# Replace these with your dataset’s left/right index pairs.
# Example: [(left_eye_idx, right_eye_idx), (left_shoulder, right_shoulder), ...]
LEFT_RIGHT_PAIRS = [(0, 1)]

def swap_keypoint_indices_(kps_on_image, pairs):
    for i, j in pairs:
        kps_on_image.keypoints[i], kps_on_image.keypoints[j] = (
            kps_on_image.keypoints[j],
            kps_on_image.keypoints[i],
        )
    return kps_on_image

aug = iaa.Fliplr(1.0)  # always flip for demo
image_aug, kps_aug = aug(image=image, keypoints=kps)

# If you apply flips probabilistically, you should swap only when a flip happened.
# For demo simplicity we always flip above.
kps_aug = kps_aug.deepcopy()
kps_aug = swap_keypoint_indices_(kps_aug, LEFT_RIGHT_PAIRS)

vis = kps_aug.draw_on_image(image_aug, size=7)
ia.imshow(vis)  # requires matplotlib
```

### If flips are probabilistic

If you use `Fliplr(0.5)`, you need to know whether a flip happened for that
sample in order to decide whether to swap indices.

Most commonly you solve this by:

- using a deterministic augmenter (`to_deterministic()`) and applying the exact
  same sampled transform to both the image and your metadata, or
- doing flips in your dataset layer (where you can easily track “did flip happen?”).

## Keep Fixed Shapes: Visibility Masks Instead of Dropping Keypoints

If your model expects a fixed number of keypoints, you generally should not
remove keypoints that go out of frame. Instead, keep them and provide a visibility
mask to the model/loss.

```python
def keypoints_visibility_mask(kps_on_image):
    h, w = kps_on_image.shape[0:2]
    return [
        (0.0 <= kp.x < w) and (0.0 <= kp.y < h)
        for kp in kps_on_image.keypoints
    ]
```

If your dataset format requires dropping keypoints, use:

- `kps_on_image.clip_out_of_image()` (drops out-of-image points)

See: [Keypoints examples](keypoints.md).

## Related

- [All Augmentables Together](all_augmentables.md)
- [Reproducibility & Determinism](../reproducibility.md)
- [Hooks](../hooks.md)

