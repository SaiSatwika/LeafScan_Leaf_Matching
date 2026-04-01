import cv2
import numpy as np


def extract_leaf(image):
    """
    Extracts the largest green object (leaf) from the image.

    Returns:
        mask: binary mask of the leaf
        leaf: cropped leaf image
    """

    if image is None:
        return None, None

    # -----------------------------
    # Convert to HSV (better for green detection)
    # -----------------------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # -----------------------------
    # Green mask (tune if needed)
    # -----------------------------
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    # -----------------------------
    # Clean mask
    # -----------------------------
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # -----------------------------
    # Find largest contour
    # -----------------------------
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    cnt = max(contours, key=cv2.contourArea)

    # Filter tiny noise
    if cv2.contourArea(cnt) < 100:
        return None, None

    # -----------------------------
    # Create clean mask
    # -----------------------------
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # -----------------------------
    # Crop leaf region
    # -----------------------------
    x, y, w, h = cv2.boundingRect(cnt)
    leaf = image[y:y+h, x:x+w]

    return clean_mask, leaf