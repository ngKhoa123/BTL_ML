# modules/preprocessing.py

import cv2
import numpy as np
import os


def preprocess_image(path, size=(224, 224)):

    img = cv2.imread(
        path,
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        return None

    mask = img > 10

    if np.any(mask):

        coords = np.column_stack(
            np.where(mask)
        )

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        img = img[
            y_min:y_max+1,
            x_min:x_max+1
        ]

    h, w = img.shape

    size_pad = max(h, w)

    padded = np.zeros(
        (size_pad, size_pad),
        dtype=img.dtype
    )

    y_offset = (size_pad - h) // 2
    x_offset = (size_pad - w) // 2

    padded[
        y_offset:y_offset+h,
        x_offset:x_offset+w
    ] = img

    img = padded

    img = cv2.resize(img, size)

    return img


def process_dataset(
    input_dir,
    output_dir
):

    for root, _, files in os.walk(input_dir):

        for file in files:

            if not file.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            in_path = os.path.join(
                root,
                file
            )

            processed = preprocess_image(
                in_path
            )

            if processed is None:
                continue

            rel_path = os.path.relpath(
                in_path,
                input_dir
            )

            out_path = os.path.join(
                output_dir,
                rel_path
            )

            os.makedirs(
                os.path.dirname(out_path),
                exist_ok=True
            )

            cv2.imwrite(
                out_path,
                processed
            )