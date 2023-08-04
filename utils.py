import numpy as np
import cv2
from typing import Any


def save_overlay_image(img_path: str, mask_path: str, prediction: Any, overlay_path: str) -> None:
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    line = np.ones((256, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    prediction = np.expand_dims(prediction, axis=-1)
    prediction = np.concatenate([prediction, prediction, prediction], axis=-1)

    overlay = np.multiply(image, prediction)
    prediction = prediction * 255

    final_img = np.concatenate([image, line, mask, line, prediction, line, overlay], axis=1)

    cv2.imwrite(overlay_path, final_img)
    return
