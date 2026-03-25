import cv2
from core.extract_leaf import extract_leaf

IMG_PATH = r"D:\GreenhouseDataset\reconstructed_defoliated\1-1-9 _ D2.jpg"

img = cv2.imread(IMG_PATH)

if img is None:
    print("❌ Image not loaded. Check path.")
    exit()

mask, leaf = extract_leaf(img)

cv2.imshow("Original", img)
cv2.imshow("Mask", mask)
cv2.imshow("Leaf", leaf)

cv2.waitKey(0)
cv2.destroyAllWindows()