# modules/utils.py

import os
import numpy as np


def save_numpy(
    array,
    save_path
):

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    np.save(
        save_path,
        array
    )

    print(f"Saved: {save_path}")