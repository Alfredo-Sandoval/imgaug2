import timeit

import imgaug2 as ia
import imgaug2.augmenters as iaa


def main():
    for size in [64, 128, 256, 512, 1024]:
        for nb_bits in [1, 2, 3, 4, 5, 6, 7, 8]:
            time_iaa = timeit.timeit(
                f"iaa.quantize_uniform_to_n_bits(image, {nb_bits})",
                number=1000,
                setup=(
                    "import imgaug2 as ia; "
                    "import imgaug2.augmenters as iaa; "
                    f"image = ia.quokka_square(({size}, {size}))")
            )
            time_pil = timeit.timeit(
                "np.asarray("
                f"PIL.ImageOps.posterize(PIL.Image.fromarray(image), {nb_bits})"
                ")",
                number=1000,
                setup=(
                    "import numpy as np; "
                    "import PIL.Image; "
                    "import PIL.ImageOps; "
                    "import imgaug2 as ia; "
                    f"image = ia.quokka_square(({size}, {size}))")
            )
            print(f"[size={size:04d}, bits={nb_bits}] iaa={time_iaa:.4f} pil={time_pil:.4f}")

    image = ia.quokka_square((128, 128))
    images_q = [iaa.quantize_uniform_to_n_bits(image, nb_bits)
                for nb_bits
                in [1, 2, 3, 4, 5, 6, 7, 8]]

    ia.imshow(ia.draw_grid(images_q, cols=8, rows=1))


def posterize(arr, n_bits):
    import numpy as np
    import PIL.Image
    import PIL.ImageOps
    img = PIL.Image.fromarray(arr)
    img_q = PIL.ImageOps.posterize(img, n_bits)
    return np.asarray(img_q)


if __name__ == "__main__":
    main()
