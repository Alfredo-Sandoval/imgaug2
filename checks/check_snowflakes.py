
import imageio

import imgaug2 as ia
from imgaug2 import augmenters as iaa


def main():
    for size in [0.1, 0.2, 1.0]:
        image = imageio.imread("https://upload.wikimedia.org/wikipedia/commons/8/89/Kukle%2CCzech_Republic..jpg",
                               format="jpg")
        image = ia.imresize_single_image(image, size, "cubic")
        print(image.shape)
        augs = [
            ("iaa.Snowflakes()", iaa.Snowflakes())
        ]

        for descr, aug in augs:
            print(descr)
            images_aug = aug.augment_images([image] * 64)
            ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
