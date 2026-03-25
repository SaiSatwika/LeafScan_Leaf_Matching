import cv2
import numpy as np
import sys


# -------------------------------------------------
# CORE FUNCTION (USED EVERYWHERE)
# -------------------------------------------------

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


# -------------------------------------------------
# VISUAL DEBUG MODE
# -------------------------------------------------

def show_preview(image, mask, leaf):

    def resize_keep_aspect(img, max_height=800):
        h, w = img.shape[:2]
        scale = max_height / h
        new_w = int(w * scale)
        return cv2.resize(img, (new_w, max_height))

    image_r = resize_keep_aspect(image)
    mask_r = resize_keep_aspect(mask)
    leaf_r = resize_keep_aspect(leaf)

    cv2.imshow("Original", image_r)
    cv2.imshow("Mask", mask_r)
    cv2.imshow("Leaf", leaf_r)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------------------------
# RUN AS SCRIPT (FOR TESTING ONLY)
# -------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python extract_leaf.py <image_path>")
        sys.exit()

    img_path = sys.argv[1]

    img = cv2.imread(img_path)

    if img is None:
        print("Invalid image path")
        sys.exit()

    mask, leaf = extract_leaf(img)

    if mask is None:
        print("Extraction failed")
        sys.exit()

    show_preview(img, mask, leaf)