from __future__ import annotations

import itertools

import numpy as np

import imgaug2.imgaug as ia
from imgaug2.augmentables.batches import UnnormalizedBatch
from imgaug2.augmenters import base as iabase
from imgaug2.augmenters._typing import Array, Images, RNGInput


class AugmenterHighLevelMixin:
    def augment(
        self,
        return_batch: bool = False,
        hooks: ia.HooksImages | None = None,
        **kwargs: object,
    ) -> object:
        """Augment a batch.

        This method is a wrapper around
        :class:`~imgaug2.augmentables.batches.UnnormalizedBatch` and
        :func:`~imgaug2.augmenters.meta.Augmenter.augment_batch`. Hence, it
        supports the same datatypes as
        :class:`~imgaug2.augmentables.batches.UnnormalizedBatch`.

        If `return_batch` was set to ``False`` (the default), the method will
        return a tuple of augmentables. It will return the same types of
        augmentables (but in augmented form) as input into the method. The
        return order matches the order of the named arguments, e.g.
        ``x_aug, y_aug, z_aug = augment(X=x, Y=y, Z=z)``.

        If `return_batch` was set to ``True``, an instance of
        :class:`~imgaug2.augmentables.batches.UnnormalizedBatch` will be
        returned.

        All augmentables must be provided as named arguments.
        E.g. ``augment(<array>)`` will crash, but ``augment(images=<array>)``
        will work.

        Parameters
        ----------
        image : None or (H,W,C) ndarray or (H,W) ndarray, optional
            The image to augment. Only this or `images` can be set, not both.

        images : None or (N,H,W,C) ndarray or (N,H,W) ndarray or iterable of (H,W,C) ndarray or iterable of (H,W) ndarray, optional
            The images to augment. Only this or `image` can be set, not both.

        heatmaps : None or (N,H,W,C) ndarray or imgaug2.augmentables.heatmaps.HeatmapsOnImage or iterable of (H,W,C) ndarray or iterable of imgaug2.augmentables.heatmaps.HeatmapsOnImage, optional
            The heatmaps to augment.
            If anything else than
            :class:`~imgaug2.augmentables.heatmaps.HeatmapsOnImage`, then the
            number of heatmaps must match the number of images provided via
            parameter `images`. The number is contained either in ``N`` or the
            first iterable's size.

        segmentation_maps : None or (N,H,W) ndarray or imgaug2.augmentables.segmaps.SegmentationMapsOnImage or iterable of (H,W) ndarray or iterable of imgaug2.augmentables.segmaps.SegmentationMapsOnImage, optional
            The segmentation maps to augment.
            If anything else than
            :class:`~imgaug2.augmentables.segmaps.SegmentationMapsOnImage`, then
            the number of segmaps must match the number of images provided via
            parameter `images`. The number is contained either in ``N`` or the
            first iterable's size.

        keypoints : None or list of (N,K,2) ndarray or tuple of number or imgaug2.augmentables.kps.Keypoint or iterable of (K,2) ndarray or iterable of tuple of number or iterable of imgaug2.augmentables.kps.Keypoint or iterable of imgaug2.augmentables.kps.KeypointOnImage or iterable of iterable of tuple of number or iterable of iterable of imgaug2.augmentables.kps.Keypoint, optional
            The keypoints to augment.
            If a tuple (or iterable(s) of tuple), then iterpreted as ``(x,y)``
            coordinates and must hence contain two numbers.
            A single tuple represents a single coordinate on one image, an
            iterable of tuples the coordinates on one image and an iterable of
            iterable of tuples the coordinates on several images. Analogous if
            :class:`~imgaug2.augmentables.kps.Keypoint` instances are used
            instead of tuples.
            If an ndarray, then ``N`` denotes the number of images and ``K``
            the number of keypoints on each image.
            If anything else than
            :class:`~imgaug2.augmentables.kps.KeypointsOnImage` is provided, then
            the number of keypoint groups must match the number of images
            provided via parameter `images`. The number is contained e.g. in
            ``N`` or in case of "iterable of iterable of tuples" in the first
            iterable's size.

        bounding_boxes : None or (N,B,4) ndarray or tuple of number or imgaug2.augmentables.bbs.BoundingBox or imgaug2.augmentables.bbs.BoundingBoxesOnImage or iterable of (B,4) ndarray or iterable of tuple of number or iterable of imgaug2.augmentables.bbs.BoundingBox or iterable of imgaug2.augmentables.bbs.BoundingBoxesOnImage or iterable of iterable of tuple of number or iterable of iterable imgaug2.augmentables.bbs.BoundingBox, optional
            The bounding boxes to augment.
            This is analogous to the `keypoints` parameter. However, each
            tuple -- and also the last index in case of arrays -- has size
            ``4``, denoting the bounding box coordinates ``x1``, ``y1``,
            ``x2`` and ``y2``.

        polygons : None or (N,#polys,#points,2) ndarray or imgaug2.augmentables.polys.Polygon or imgaug2.augmentables.polys.PolygonsOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug2.augmentables.kps.Keypoint or iterable of imgaug2.augmentables.polys.Polygon or iterable of imgaug2.augmentables.polys.PolygonsOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug2.augmentables.kps.Keypoint or iterable of iterable of imgaug2.augmentables.polys.Polygon or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug2.augmentables.kps.Keypoint, optional
            The polygons to augment.
            This is similar to the `keypoints` parameter. However, each polygon
            may be made up of several ``(x,y) ``coordinates (three or more are
            required for valid polygons).
            The following datatypes will be interpreted as a single polygon on
            a single image:

              * ``imgaug2.augmentables.polys.Polygon``
              * ``iterable of tuple of number``
              * ``iterable of imgaug2.augmentables.kps.Keypoint``

            The following datatypes will be interpreted as multiple polygons
            on a single image:

              * ``imgaug2.augmentables.polys.PolygonsOnImage``
              * ``iterable of imgaug2.augmentables.polys.Polygon``
              * ``iterable of iterable of tuple of number``
              * ``iterable of iterable of imgaug2.augmentables.kps.Keypoint``
              * ``iterable of iterable of imgaug2.augmentables.polys.Polygon``

            The following datatypes will be interpreted as multiple polygons on
            multiple images:

              * ``(N,#polys,#points,2) ndarray``
              * ``iterable of (#polys,#points,2) ndarray``
              * ``iterable of iterable of (#points,2) ndarray``
              * ``iterable of iterable of iterable of tuple of number``
              * ``iterable of iterable of iterable of tuple of imgaug2.augmentables.kps.Keypoint``

        line_strings : None or (N,#lines,#points,2) ndarray or imgaug2.augmentables.lines.LineString or imgaug2.augmentables.lines.LineStringOnImage or iterable of (#polys,#points,2) ndarray or iterable of tuple of number or iterable of imgaug2.augmentables.kps.Keypoint or iterable of imgaug2.augmentables.lines.LineString or iterable of imgaug2.augmentables.lines.LineStringOnImage or iterable of iterable of (#points,2) ndarray or iterable of iterable of tuple of number or iterable of iterable of imgaug2.augmentables.kps.Keypoint or iterable of iterable of imgaug2.augmentables.lines.LineString or iterable of iterable of iterable of tuple of number or iterable of iterable of iterable of tuple of imgaug2.augmentables.kps.Keypoint, optional
            The line strings to augment.
            See `polygons`, which behaves similarly.

        return_batch : bool, optional
            Whether to return an instance of
            :class:`~imgaug2.augmentables.batches.UnnormalizedBatch`.

        hooks : None or imgaug2.imgaug2.HooksImages, optional
            Hooks object to dynamically interfere with the augmentation process.

        Returns
        -------
        tuple or imgaug2.augmentables.batches.UnnormalizedBatch
            If `return_batch` was set to ``True``, a instance of
            ``UnnormalizedBatch`` will be returned.
            If `return_batch` was set to ``False``, a tuple of augmentables
            will be returned, e.g. ``(augmented images, augmented keypoints)``.
            The datatypes match the input datatypes of the corresponding named
            arguments. The order matches the order of the named arguments.

        Examples
        --------
        >>> import numpy as np
        >>> import imgaug2 as ia
        >>> import imgaug2.augmenters as iaa
        >>> aug = iaa.Affine(rotate=(-25, 25))
        >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
        >>> keypoints = [(10, 20), (30, 32)]  # (x,y) coordinates
        >>> images_aug, keypoints_aug = aug.augment(
        >>>     image=image, keypoints=keypoints)

        Create a single image and a set of two keypoints on it, then
        augment both by applying a random rotation between ``-25`` deg and
        ``+25`` deg. The sampled rotation value is automatically aligned
        between image and keypoints.

        >>> import numpy as np
        >>> import imgaug2 as ia
        >>> import imgaug2.augmenters as iaa
        >>> from imgaug2.augmentables.bbs import BoundingBox
        >>> aug = iaa.Affine(rotate=(-25, 25))
        >>> images = [np.zeros((64, 64, 3), dtype=np.uint8),
        >>>           np.zeros((32, 32, 3), dtype=np.uint8)]
        >>> keypoints = [[(10, 20), (30, 32)],  # KPs on first image
        >>>              [(22, 10), (12, 14)]]  # KPs on second image
        >>> bbs = [
        >>>           [BoundingBox(x1=5, y1=5, x2=50, y2=45)],
        >>>           [BoundingBox(x1=4, y1=6, x2=10, y2=15),
        >>>            BoundingBox(x1=8, y1=9, x2=16, y2=30)]
        >>>       ]  # one BB on first image, two BBs on second image
        >>> batch_aug = aug.augment(
        >>>     images=images, keypoints=keypoints, bounding_boxes=bbs,
        >>>     return_batch=True)

        Create two images of size ``64x64`` and ``32x32``, two sets of
        keypoints (each containing two keypoints) and two sets of bounding
        boxes (the first containing one bounding box, the second two bounding
        boxes). These augmentables are then augmented by applying random
        rotations between ``-25`` deg and ``+25`` deg to them. The rotation
        values are sampled by image and aligned between all augmentables on
        the same image. The method finally returns an instance of
        :class:`~imgaug2.augmentables.batches.UnnormalizedBatch` from which the
        augmented data can be retrieved via ``batch_aug.images_aug``,
        ``batch_aug.keypoints_aug``, and ``batch_aug.bounding_boxes_aug``.
        The augmented data can be retrieved as
        ``images_aug, keypoints_aug, bbs_aug = augment(...)``.

        """
        assert ia.is_single_bool(return_batch), (
            f"Expected boolean as argument for 'return_batch', got type {str(type(return_batch))}. "
            "Call augment() only with named arguments, e.g. "
            "augment(images=<array>)."
        )

        expected_keys = [
            "images",
            "heatmaps",
            "segmentation_maps",
            "keypoints",
            "bounding_boxes",
            "polygons",
            "line_strings",
        ]
        expected_keys_call = ["image"] + expected_keys

        # at least one augmentable provided?
        assert any([key in kwargs for key in expected_keys_call]), (
            "Expected augment() to be called with one of the following named "
            "arguments: {}. Got none of these.".format(
                ", ".join(expected_keys_call),
            )
        )

        # all keys in kwargs actually known?
        unknown_args = [key for key in kwargs if key not in expected_keys_call]
        assert len(unknown_args) == 0, (
            "Got the following unknown keyword argument(s) in augment(): {}".format(
                ", ".join(unknown_args)
            )
        )

        # normalize image=... input to images=...
        # this is not done by Batch.to_normalized_batch()
        if "image" in kwargs:
            assert "images" not in kwargs, (
                "You may only provide the argument 'image' OR 'images' to "
                "augment(), not both of them."
            )
            images = [kwargs["image"]]
            iabase._warn_on_suspicious_single_image_shape(images[0])
        else:
            images = kwargs.get("images", None)
            iabase._warn_on_suspicious_multi_image_shapes(images)

        # Python 3.7+ preserves kwargs order, which is guaranteed for 3.9+.
        order = "kwargs_keys"

        # augment batch
        batch = UnnormalizedBatch(
            images=images,
            heatmaps=kwargs.get("heatmaps", None),
            segmentation_maps=kwargs.get("segmentation_maps", None),
            keypoints=kwargs.get("keypoints", None),
            bounding_boxes=kwargs.get("bounding_boxes", None),
            polygons=kwargs.get("polygons", None),
            line_strings=kwargs.get("line_strings", None),
        )

        batch_aug = self.augment_batch_(batch, hooks=hooks)

        # return either batch or tuple of augmentables, depending on what
        # was requested by user
        if return_batch:
            return batch_aug

        result = []
        if order == "kwargs_keys":
            for key in kwargs:
                if key == "image":
                    attr = batch_aug.images_aug
                    result.append(attr[0])
                else:
                    result.append(getattr(batch_aug, f"{key}_aug"))
        else:
            for key in expected_keys:
                if key == "images" and "image" in kwargs:
                    attr = batch_aug.images_aug
                    result.append(attr[0])
                elif key in kwargs:
                    result.append(getattr(batch_aug, f"{key}_aug"))

        if len(result) == 1:
            return result[0]
        return tuple(result)

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Alias for :func:`~imgaug2.augmenters.meta.Augmenter.augment`."""
        return self.augment(*args, **kwargs)

    def pool(
        self,
        processes: int | None = None,
        maxtasksperchild: int | None = None,
        seed: RNGInput = None,
    ) -> object:
        """Create a pool used for multicore augmentation.

        Parameters
        ----------
        processes : None or int, optional
            Same as in :func:`~imgaug2.multicore.Pool.__init__`.
            The number of background workers. If ``None``, the number of the
            machine's CPU cores will be used (this counts hyperthreads as CPU
            cores). If this is set to a negative value ``p``, then
            ``P - abs(p)`` will be used, where ``P`` is the number of CPU
            cores. E.g. ``-1`` would use all cores except one (this is useful
            to e.g. reserve one core to feed batches to the GPU).

        maxtasksperchild : None or int, optional
            Same as for :func:`~imgaug2.multicore.Pool.__init__`.
            The number of tasks done per worker process before the process
            is killed and restarted. If ``None``, worker processes will not
            be automatically restarted.

        seed : None or int, optional
            Same as for :func:`~imgaug2.multicore.Pool.__init__`.
            The seed to use for child processes. If ``None``, a random seed
            will be used.

        Returns
        -------
        imgaug2.multicore.Pool
            Pool for multicore augmentation.

        Examples
        --------
        >>> import numpy as np
        >>> import imgaug2 as ia
        >>> import imgaug2.augmenters as iaa
        >>> from imgaug2.augmentables.batches import Batch
        >>>
        >>> aug = iaa.Add(1)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> batches = [Batch(images=np.copy(images)) for _ in range(100)]
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.map_batches(batches, chunksize=8)
        >>> print(np.sum(batches_aug[0].images_aug[0]))
        49152

        Create ``100`` batches of empty images. Each batch contains
        ``16`` images of size ``128x128``. The batches are then augmented on
        all CPU cores except one (``processes=-1``). After augmentation, the
        sum of pixel values from the first augmented image is printed.

        >>> import numpy as np
        >>> import imgaug2 as ia
        >>> import imgaug2.augmenters as iaa
        >>> from imgaug2.augmentables.batches import Batch
        >>>
        >>> aug = iaa.Add(1)
        >>> images = np.zeros((16, 128, 128, 3), dtype=np.uint8)
        >>> def generate_batches():
        >>>     for _ in range(100):
        >>>         yield Batch(images=np.copy(images))
        >>>
        >>> with aug.pool(processes=-1, seed=2) as pool:
        >>>     batches_aug = pool.imap_batches(generate_batches(), chunksize=8)
        >>>     batch_aug = next(batches_aug)
        >>>     print(np.sum(batch_aug.images_aug[0]))
        49152

        Same as above. This time, a generator is used to generate batches
        of images. Again, the first augmented image's sum of pixels is printed.

        """
        import imgaug2.multicore as multicore

        return multicore.Pool(
            self, processes=processes, maxtasksperchild=maxtasksperchild, seed=seed
        )

    # TODO most of the code of this function could be replaced with
    #      ia.draw_grid()
    # TODO add parameter for handling multiple images ((a) next to each other
    #      in each row or (b) multiply row count by number of images and put
    #      each one in a new row)
    # TODO "images" parameter deviates from augment_images (3d array is here
    #      treated as one 3d image, in augment_images as (N, H, W))
    # TODO according to the docstring, this can handle (H,W) images, but not
    #      (H,W,1)
    def draw_grid(self, images: Images, rows: int, cols: int) -> Array:
        """Augment images and draw the results as a single grid-like image.

        This method applies this augmenter to the provided images and returns
        a grid image of the results. Each cell in the grid contains a single
        augmented version of an input image.

        If multiple input images are provided, the row count is multiplied by
        the number of images and each image gets its own row.
        E.g. for ``images = [A, B]``, ``rows=2``, ``cols=3``::

            A A A
            B B B
            A A A
            B B B

        for ``images = [A]``, ``rows=2``, ``cols=3``::

            A A A
            A A A

        Parameters
        -------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images to augment and draw in the grid.
            If a list, then each element is expected to have shape ``(H, W)``
            or ``(H, W, 3)``. If a single array, then it is expected to have
            shape ``(N, H, W, 3)`` or ``(H, W, 3)`` or ``(H, W)``.

        rows : int
            Number of rows in the grid.
            If ``N`` input images are given, this value will automatically be
            multiplied by ``N`` to create rows for each image.

        cols : int
            Number of columns in the grid.

        Returns
        -------
        (Hg, Wg, 3) ndarray
            The generated grid image with augmented versions of the input
            images. Here, ``Hg`` and ``Wg`` reference the output size of the
            grid, and *not* the sizes of the input images.

        """
        if ia.is_np_array(images):
            if len(images.shape) == 4:
                images = [images[i] for i in range(images.shape[0])]
            elif len(images.shape) == 3:
                images = [images]
            elif len(images.shape) == 2:
                images = [images[:, :, np.newaxis]]
            else:
                raise Exception(
                    "Unexpected images shape, expected 2-, 3- or "
                    f"4-dimensional array, got shape {images.shape}."
                )
        else:
            assert isinstance(images, list), (
                f"Expected 'images' to be an ndarray or list of ndarrays. Got {type(images)}."
            )
            for i, image in enumerate(images):
                if len(image.shape) == 3:
                    continue
                if len(image.shape) == 2:
                    images[i] = image[:, :, np.newaxis]
                else:
                    raise Exception(
                        f"Unexpected image shape at index {i}, expected 2- or "
                        f"3-dimensional array, got shape {image.shape}."
                    )

        det = self if self.deterministic else self.to_deterministic()
        augs = []
        for image in images:
            augs.append(det.augment_images([image] * (rows * cols)))

        augs_flat = list(itertools.chain(*augs))
        cell_height = max([image.shape[0] for image in augs_flat])
        cell_width = max([image.shape[1] for image in augs_flat])
        width = cell_width * cols
        height = cell_height * (rows * len(images))
        grid = np.zeros((height, width, 3), dtype=augs[0][0].dtype)
        for row_idx in range(rows):
            for img_idx, _image in enumerate(images):
                for col_idx in range(cols):
                    image_aug = augs[img_idx][(row_idx * cols) + col_idx]
                    cell_y1 = cell_height * (row_idx * len(images) + img_idx)
                    cell_y2 = cell_y1 + image_aug.shape[0]
                    cell_x1 = cell_width * col_idx
                    cell_x2 = cell_x1 + image_aug.shape[1]
                    grid[cell_y1:cell_y2, cell_x1:cell_x2, :] = image_aug

        return grid

    def show_grid(self, images: Images, rows: int, cols: int) -> None:
        """Augment images and plot the results as a single grid-like image.

        This calls :func:`~imgaug2.augmenters.meta.Augmenter.draw_grid` and
        simply shows the results. See that method for details.

        Parameters
        ----------
        images : (N,H,W,3) ndarray or (H,W,3) ndarray or (H,W) ndarray or list of (H,W,3) ndarray or list of (H,W) ndarray
            List of images to augment and draw in the grid.
            If a list, then each element is expected to have shape ``(H, W)``
            or ``(H, W, 3)``. If a single array, then it is expected to have
            shape ``(N, H, W, 3)`` or ``(H, W, 3)`` or ``(H, W)``.

        rows : int
            Number of rows in the grid.
            If ``N`` input images are given, this value will automatically be
            multiplied by ``N`` to create rows for each image.

        cols : int
            Number of columns in the grid.

        """
        grid = self.draw_grid(images, rows, cols)
        ia.imshow(grid)

