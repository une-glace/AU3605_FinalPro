import cv2
import numpy as np
import os

def color_normalization(test_image):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.join(current_dir, "ref_img.jpg")
    
    ref_image = cv2.imread(ref_path)
    if ref_image is None:
        raise FileNotFoundError(f"Reference image not found at {ref_path}")
        
    test_image = test_image.astype(np.float32)
    test_image = test_image.copy()
    ref_image = ref_image.astype(np.float32)

    test_mean, test_std = cv2.meanStdDev(test_image)
    ref_mean, ref_std = cv2.meanStdDev(ref_image)

    normalized_image = np.zeros_like(test_image)
    for i in range(3):
        if test_std[i] == 0:
            normalized_image[:, :, i] = test_image[:, :, i]
        else:
            normalized_image[:, :, i] = (
                (test_image[:, :, i] - test_mean[i]) / test_std[i]
            ) * ref_std[i] + ref_mean[i]

    normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)

    return normalized_image

