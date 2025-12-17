import imgaug2 as ia
import imgaug2.augmenters as iaa


def main():
    image = ia.quokka_square((128, 128))
    aug = iaa.MeanShiftBlur()
    images_aug = aug(images=[image] * 16)
    ia.imshow(ia.draw_grid(images_aug))


if __name__ == "__main__":
    main()
