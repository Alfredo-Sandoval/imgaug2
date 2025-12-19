import timeit

import imgaug2 as ia
import imgaug2.augmenters as iaa


def main():
    for size in [64, 128, 256, 512, 1024]:
        for threshold in [64, 128, 192]:
            time_iaa = timeit.timeit(
                f"iaa.solarize(image, {threshold})",
                number=1000,
                setup=(
                    "import imgaug2 as ia; "
                    "import imgaug2.augmenters as iaa; "
                    f"image = ia.quokka_square(({size}, {size}))")
            )
            time_pil = timeit.timeit(
                "np.asarray("
                f"PIL.ImageOps.solarize(PIL.Image.fromarray(image), {threshold})"
                ")",
                number=1000,
                setup=(
                    "import numpy as np; "
                    "import PIL.Image; "
                    "import PIL.ImageOps; "
                    "import imgaug2 as ia; "
                    f"image = ia.quokka_square(({size}, {size}))")
            )
            print(f"[size={size:04d}, thresh={threshold:03d}] iaa={time_iaa:.4f} pil={time_pil:.4f}")

    image = ia.quokka_square((128, 128))
    images_aug = iaa.Solarize(1.0)(images=[image] * (5*5))
    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
