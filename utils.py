import numpy as np
import cv2

def save_overlay_image(image_path, mask_path, prediction, overlay_path):
    line = np.ones((256, 10, 3)) * 128

    input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    ground_truth_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    ground_truth_mask = np.expand_dims(ground_truth_mask, axis=-1)
    ground_truth_mask = np.concatenate(
        [ground_truth_mask, ground_truth_mask, ground_truth_mask], axis=-1
    )

    prediction = np.expand_dims(prediction, axis=-1)
    prediction = np.concatenate([prediction, prediction, prediction], axis=-1)

    overlay = input_image * prediction
    prediction = prediction * 255

    final_img = np.concatenate(
        [input_image, line, ground_truth_mask, line, prediction, line, overlay], axis=1
    )

    cv2.imwrite(overlay_path, final_img)
    return