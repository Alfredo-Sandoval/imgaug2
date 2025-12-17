# Basic Examples

This page shows common imgaug2 workflows from simple to advanced.

## A Simple Augmentation Pipeline

```python
import numpy as np
import imgaug2 as ia
import imgaug2.augmenters as iaa

# Create a simple augmentation sequence
seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),  # Crop images by 0-16px
    iaa.Fliplr(0.5),       # Horizontal flip 50% of images
    iaa.GaussianBlur(sigma=(0, 3.0))  # Blur with sigma 0-3
])

# Load sample images
images = [ia.quokka(size=(256, 256)) for _ in range(16)]

# Augment the batch
images_aug = seq(images=images)
```

## A Common Augmentation Sequence

This sequence combines multiple augmentation types applied in random order:

```python
import imgaug2.augmenters as iaa

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Horizontal flips
    iaa.Crop(percent=(0, 0.1)),  # Random crops

    # Apply 0-5 of the following augmenters
    iaa.SomeOf((0, 5), [
        # Blur
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),

        # Sharpen or emboss
        iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
        iaa.Sometimes(0.5, iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))),

        # Add noise
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(
            loc=0, scale=(0.0, 0.05*255), per_channel=0.5
        )),

        # Dropout
        iaa.Sometimes(0.5, iaa.OneOf([
            iaa.Dropout((0.01, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
        ])),

        # Invert colors
        iaa.Sometimes(0.5, iaa.Invert(0.05, per_channel=True)),

        # Brightness and contrast
        iaa.Sometimes(0.5, iaa.Add((-10, 10), per_channel=0.5)),
        iaa.Sometimes(0.5, iaa.Multiply((0.5, 1.5), per_channel=0.5)),
        iaa.Sometimes(0.5, iaa.LinearContrast((0.5, 2.0), per_channel=0.5)),

        # Grayscale
        iaa.Sometimes(0.5, iaa.Grayscale(alpha=(0.0, 1.0))),

        # Geometric
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )),

        # Perspective
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.1))),
    ], random_order=True)
], random_order=True)

# Apply to images
images_aug = seq(images=images)
```

## Heavy Augmentations

For aggressive data augmentation (may need tuning):

```python
import imgaug2.augmenters as iaa

# Warning: This pipeline is very aggressive
seq = iaa.Sequential([
    iaa.Sometimes(0.5, iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
    iaa.OneOf([
        iaa.GaussianBlur((0, 3.0)),
        iaa.AverageBlur(k=(2, 7)),
        iaa.MedianBlur(k=(3, 11)),
    ]),
    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.OneOf([
        iaa.Dropout((0.01, 0.1), per_channel=0.5),
        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
    ]),
    iaa.Invert(0.05, per_channel=True),
    iaa.Add((-10, 10), per_channel=0.5),
    iaa.Multiply((0.5, 1.5), per_channel=0.5),
    iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
    iaa.Grayscale(alpha=(0.0, 1.0)),
    iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),
    iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-45, 45),
        shear=(-16, 16),
    ),
], random_order=True)
```

!!! warning
    The heavy augmentation pipeline above may be too aggressive for many use cases.
    Consider reducing the parameter ranges or wrapping augmenters in `Sometimes()`.

## Using with Training Loops

```python
import imgaug2.augmenters as iaa

# Define augmentation once
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-20, 20)),
    iaa.GaussianBlur(sigma=(0, 1.0))
])

# In your training loop
for epoch in range(num_epochs):
    for batch_images, batch_labels in dataloader:
        # Augment each batch
        batch_images_aug = aug(images=batch_images)

        # Train on augmented images
        loss = model.train_step(batch_images_aug, batch_labels)
```

## Reproducibility with Seeds

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa

# Set global seed
ia.seed(42)

# Or use deterministic augmenters
aug = iaa.Affine(rotate=(-20, 20))
aug_det = aug.to_deterministic()

# Same augmentation applied to all inputs
image1_aug = aug_det(image=image1)
image2_aug = aug_det(image=image2)  # Different augmentation

# For same augmentation, create new deterministic augmenter
aug_det2 = aug.to_deterministic()
image3_aug = aug_det2(image=image1)  # Same as image1_aug? No, new random state
```

## Visualizing Augmentations

```python
import imgaug2 as ia
import imgaug2.augmenters as iaa

# Load a sample image
image = ia.quokka(size=(256, 256))

# Create augmenter
aug = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 2.0)),
    iaa.Affine(rotate=(-25, 25))
])

# Generate multiple augmented versions
images_aug = [aug(image=image) for _ in range(16)]

# Create a grid visualization
grid = ia.draw_grid(images_aug, cols=4)

# Save or display
import imageio
imageio.imwrite("augmented_grid.jpg", grid)
```
