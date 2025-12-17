import imgaug2 as ia
from imgaug2 import augmenters as iaa


def main():
    img = ia.data.quokka(0.5)
    augs = [
        ("iaa.AdditivePoissonNoise(0)", iaa.AdditivePoissonNoise(0)),
        ("iaa.AdditivePoissonNoise(10.0)", iaa.AdditivePoissonNoise(10.0)),
        ("iaa.AdditivePoissonNoise(20.0)", iaa.AdditivePoissonNoise(20.0)),
        ("iaa.AdditivePoissonNoise(50.0)", iaa.AdditivePoissonNoise(50.0)),
        ("iaa.AdditivePoissonNoise((10.0, 20))", iaa.AdditivePoissonNoise((10.0, 20))),
        ("iaa.AdditivePoissonNoise([10.0, 20.0, 50])", iaa.AdditivePoissonNoise([10.0, 20.0, 50])),
        ("iaa.AdditivePoissonNoise(20, per_channel=True)", iaa.AdditivePoissonNoise(50, per_channel=True)),
    ]
    for descr, aug in augs:
        print(descr)
        imgs_aug = aug.augment_images([img] * 16)
        ia.imshow(ia.draw_grid(imgs_aug))


if __name__ == "__main__":
    main()
