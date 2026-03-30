import cv2
import numpy as np

def extract_leaf(image):

    if image is None:
        return None, None

    # Detect non-black pixels
    mask = np.any(image > 5, axis=2).astype(np.uint8) * 255

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep largest component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    if num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest).astype(np.uint8) * 255

    # Apply mask
    leaf = cv2.bitwise_and(image, image, mask=mask)

    return mask, leaf