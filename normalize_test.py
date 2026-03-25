import cv2
from core.extract_leaf import extract_leaf
from normalize_leaf import normalize_leaf

IMG_PATH = r"D:\GreenhouseDataset\reconstructed_defoliated\1-1-9 _ D2.jpg"

img = cv2.imread(IMG_PATH)

mask, leaf = extract_leaf(img)
mask_norm, leaf_norm = normalize_leaf(mask, leaf)

# -----------------------------
# DISPLAY FIX (IMPORTANT)
# -----------------------------
def resize_for_display(image, max_height=800):
    h, w = image.shape[:2]

    if h > max_height:
        scale = max_height / h
        new_w = int(w * scale)
        image = cv2.resize(image, (new_w, max_height))

    return image

cv2.imshow("Original", resize_for_display(img))
cv2.imshow("Leaf", resize_for_display(leaf))
cv2.imshow("Normalized", resize_for_display(leaf_norm))

cv2.waitKey(0)
cv2.destroyAllWindows()