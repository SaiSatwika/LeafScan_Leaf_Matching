import cv2
import numpy as np
from pathlib import Path
import re

# -----------------------------
# PATHS
# -----------------------------
HEALTHY_DIR = Path(r"D:\GreenhouseDataset\reconstructed_healthy")
DEFOLIATED_DIR = Path(r"D:\GreenhouseDataset\reconstructed_defoliated")

# -----------------------------
# MASK FUNCTIONS
# -----------------------------
def get_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    return mask

def smooth_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.medianBlur(mask, 5)
    return mask

def compute_area(mask):
    return np.sum(mask > 0)

# -----------------------------
# STRONG KEY EXTRACTION (FIXED)
# -----------------------------
def extract_key(name):
    name = name.replace(".jpg", "").replace(".png", "").strip()

    # remove ALL spaces around underscores
    name = re.sub(r"\s*_\s*", "_", name)

    # ---------- HEALTHY ----------
    # Example: "1-1_Leaf9" or "1-1_Leaf 9"
    if "_Leaf" in name or "_leaf" in name:
        try:
            plant, leaf_part = name.split("_")
            leaf_num = "".join(filter(str.isdigit, leaf_part))
            return f"{plant}-{leaf_num}"
        except:
            return None

    # ---------- DEFOLIATED ----------
    # Example: "1-1-9_D2"
    if "_D" in name or "_d" in name:
        try:
            base = name.split("_")[0]
            return base
        except:
            return None

    return None

# -----------------------------
# LOAD FILES
# -----------------------------
healthy_files = list(HEALTHY_DIR.glob("*.*"))
defoliated_files = list(DEFOLIATED_DIR.glob("*.*"))

print(f"Healthy images: {len(healthy_files)}")
print(f"Defoliated images: {len(defoliated_files)}")

# -----------------------------
# DEBUG KEYS
# -----------------------------
print("\n--- SAMPLE KEYS ---")

for f in healthy_files[:5]:
    print("Healthy:", f.name, "→", extract_key(f.name))

for f in defoliated_files[:5]:
    print("Defoliated:", f.name, "→", extract_key(f.name))

# -----------------------------
# CREATE HEALTHY MAP
# -----------------------------
healthy_map = {}

for f in healthy_files:
    key = extract_key(f.name)
    if key is not None:
        healthy_map[key] = f

print(f"\nValid healthy keys: {len(healthy_map)}")

# -----------------------------
# ANALYSIS
# -----------------------------
results = []
skipped = 0

for dfile in defoliated_files:

    key = extract_key(dfile.name)

    if key is None or key not in healthy_map:
        skipped += 1
        continue

    hfile = healthy_map[key]

    img_h = cv2.imread(str(hfile))
    img_d = cv2.imread(str(dfile))

    if img_h is None or img_d is None:
        skipped += 1
        continue

    # RAW masks
    mask_h_raw = get_mask(img_h)
    mask_d_raw = get_mask(img_d)

    # SMOOTHED masks
    mask_h_smooth = smooth_mask(mask_h_raw)
    mask_d_smooth = smooth_mask(mask_d_raw)

    # Areas
    h_raw = compute_area(mask_h_raw)
    d_raw = compute_area(mask_d_raw)

    h_smooth = compute_area(mask_h_smooth)
    d_smooth = compute_area(mask_d_smooth)

    eps = 1e-6

    def_raw = (h_raw - d_raw) / (h_raw + eps)
    def_smooth = (h_smooth - d_smooth) / (h_smooth + eps)

    results.append({
        "leaf": key,
        "def_raw": def_raw,
        "def_smooth": def_smooth,
        "diff": abs(def_raw - def_smooth)
    })

# -----------------------------
# RESULTS
# -----------------------------
print("\n===== RESULTS =====")
print(f"Samples matched: {len(results)}")
print(f"Skipped (no match): {skipped}")

if len(results) == 0:
    print("\n❌ Still no matches — but this should not happen now")
else:
    diffs = [r["diff"] for r in results]

    print(f"\nAverage defoliation difference: {np.mean(diffs):.4f}")
    print(f"Max difference: {np.max(diffs):.4f}")

    print("\nTop 5 worst cases:")
    for r in sorted(results, key=lambda x: x["diff"], reverse=True)[:5]:
        print(r)