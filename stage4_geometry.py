import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from core.extract_leaf import extract_leaf

# -------------------------------------------------
# PATHS
# -------------------------------------------------

HEALTHY_DIR = Path(r"D:\updated dataset\Healthy_reconstruction")
DEFOLIATED_DIR = Path(r"D:\updated dataset\Defoliated_reconstruction")

OUT_PATH = Path("outputs/features")
OUT_PATH.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# FEATURE FUNCTION (same logic)
# -------------------------------------------------

def compute_geometry_features(mask):

    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    convexity = area / hull_area if hull_area > 0 else 0.0
    compactness = area / (perimeter ** 2) if perimeter > 0 else 0.0

    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean, eigvecs, eigvals = cv2.PCACompute2(pts, mean=None)

    major_axis = 2 * np.sqrt(eigvals[0][0])
    minor_axis = 2 * np.sqrt(eigvals[1][0])
    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0

    return {
        "area": area,
        "perimeter": perimeter,
        "hull_area": hull_area,
        "convexity": convexity,
        "compactness": compactness,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "aspect_ratio": aspect_ratio,
    }

# -------------------------------------------------
# PROCESS FUNCTION
# -------------------------------------------------

def process_folder(folder, category):

    records = []

    image_paths = sorted(list(folder.glob("*.*")))

    print(f"\nProcessing {category}: {len(image_paths)} images")

    for img_path in image_paths:

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        mask, leaf = extract_leaf(img)

        if mask is None:
            continue

        feats = compute_geometry_features(mask)
        if feats is None:
            continue

        leaf_id = img_path.stem

        row = {
            "leaf_id": leaf_id,
            "category": category,
            "is_healthy": 1 if category == "healthy" else 0,
        }

        row.update(feats)
        records.append(row)

    return records

# -------------------------------------------------
# MAIN
# -------------------------------------------------

rows = []

rows += process_folder(HEALTHY_DIR, "healthy")
rows += process_folder(DEFOLIATED_DIR, "simulated")

df = pd.DataFrame(rows)

df.to_csv(OUT_PATH / "phase1_geometry.csv", index=False)

print("\n✅ Geometry extraction complete")
print("Total samples:", len(df))
print(df.head())