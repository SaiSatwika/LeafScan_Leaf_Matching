import cv2
import numpy as np

# -------------------------------------------------
# PCA ANGLE (unchanged)
# -------------------------------------------------

def compute_pca_angle(mask):

    ys, xs = np.where(mask > 0)

    if len(xs) < 50:
        return 0.0

    coords = np.column_stack((xs, ys)).astype(np.float32)

    mean = np.mean(coords, axis=0)
    coords -= mean

    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)

    major_axis = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(major_axis[1], major_axis[0]))

    return angle - 90


# -------------------------------------------------
# ROTATION (unchanged)
# -------------------------------------------------

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderValue=0
    )

    return rotated


# -------------------------------------------------
# TIGHT CROP (unchanged)
# -------------------------------------------------

def tight_crop(mask, image=None):

    coords = cv2.findNonZero(mask)
    if coords is None:
        return mask, image

    x, y, w, h = cv2.boundingRect(coords)

    cropped_mask = mask[y:y+h, x:x+w]
    cropped_img = None

    if image is not None:
        cropped_img = image[y:y+h, x:x+w]

    return cropped_mask, cropped_img


# -------------------------------------------------
# NORMALIZATION
# -------------------------------------------------

def normalize_leaf(mask, image=None):

    # Step 1: PCA rotation
    angle = compute_pca_angle(mask)

    mask_rot = rotate_image(mask, angle)
    img_rot = rotate_image(image, angle) if image is not None else None

    # Step 2: tight crop
    mask_norm, img_norm = tight_crop(mask_rot, img_rot)

    return mask_norm, img_norm