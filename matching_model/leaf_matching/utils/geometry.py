import cv2
import numpy as np

def compute_geometry(mask):

    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))

    convexity = float(area / hull_area) if hull_area > 0 else 0.0
    compactness = float(area / (perimeter ** 2)) if perimeter > 0 else 0.0

    pts = cnt.reshape(-1, 2).astype(np.float32)
    _, eigvecs, eigvals = cv2.PCACompute2(pts, mean=None)

    major_axis = float(2 * np.sqrt(eigvals[0][0]))
    minor_axis = float(2 * np.sqrt(eigvals[1][0]))

    return {
        "area": area,
        "perimeter": perimeter,
        "convexity": convexity,
        "compactness": compactness,
        "major_axis": major_axis,
        "minor_axis": minor_axis
    }