
import numpy as np

import imgaug2 as ia


def main():
    image = ia.data.quokka()
    image_gray = np.average(image, axis=2).astype(np.uint8)
    image_gray_3d = image_gray[:, :, np.newaxis]

    ia.imshow(image)
    ia.imshow(image / 255.0)
    ia.imshow(image_gray)
    ia.imshow(image_gray_3d)


if __name__ == "__main__":
    main()
