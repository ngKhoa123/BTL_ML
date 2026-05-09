# modules/feature_extraction.py

import numpy as np

from PIL import Image

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import hog


IMG_SIZE = (128, 128)

ORIENTATIONS = 12

PIXELS_PER_CELL = (8, 8)

CELLS_PER_BLOCK = (2, 2)


def extract_hog(path):

    img = Image.open(path).convert(
        "RGB"
    )

    img = np.array(img)

    img = rgb2gray(img)

    img = resize(
        img,
        IMG_SIZE,
        anti_aliasing=True
    )

    features = hog(
        img,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True
    )

    return features