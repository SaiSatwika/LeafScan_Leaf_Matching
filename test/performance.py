import sys
from pathlib import Path

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

GT_PATH = PROJECT_ROOT / "outputs/features/gt_mapping.csv"
TEST_IMAGE_DIR = Path(r"D:\Final_dataset\Defoliated") 

# -----------------------------
# LOAD GT
# -----------------------------

df_gt = pd.read_csv(GT_PATH)

# -----------------------------
# FIXED GT MAPPING
# -----------------------------

def extract_key_from_filename(filename):

    nums = re.findall(r'\d+', filename)

    # Try removing last number (scan/frame)
    if len(nums) > 4:
        key = "-".join(nums[:-1])
        if key in df_gt["leaf_key"].values:
            return key

    # fallback: full key
    key_full = "-".join(nums)
    if key_full in df_gt["leaf_key"].values:
        return key_full

    return None

# -----------------------------
# API CONFIG
# -----------------------------

url = "http://127.0.0.1:5000/predict"

# -----------------------------
# LOAD IMAGES (ALL FORMATS)
# -----------------------------

images = []
for ext in ["*.jpg", "*.png", "*.jpeg"]:
    images.extend(TEST_IMAGE_DIR.glob(ext))

print(f"\n🔍 Found {len(images)} images")

# -----------------------------
# RUN EVALUATION
# -----------------------------

preds = []
actuals = []

total = 0
matched = 0

print("\n🚀 Running evaluation...\n")

for img_path in tqdm(images):

    total += 1

    leaf_key = extract_key_from_filename(img_path.stem)

    if leaf_key is None:
        print("❌ No GT match:", img_path.name)
        continue

    gt_row = df_gt[df_gt["leaf_key"] == leaf_key]

    if len(gt_row) == 0:
        print("❌ GT missing:", leaf_key)
        continue

    matched += 1

    actual = float(gt_row["defoliation_gt"].values[0])

    try:
        with open(img_path, "rb") as f:
            response = requests.post(url, files={"image": f})

        data = response.json()
        pred = float(data.get("prediction"))

    except:
        print("❌ API failed:", img_path.name)
        continue

    preds.append(pred)
    actuals.append(actual)

# -----------------------------
# SUMMARY
# -----------------------------

print(f"\n📊 GT matched: {matched}/{total}")

preds = np.array(preds)
actuals = np.array(actuals)

if len(preds) == 0:
    print("❌ No valid predictions — check paths or API")
    exit()

# -----------------------------
# METRICS
# -----------------------------

mae = np.mean(np.abs(preds - actuals))
rmse = np.sqrt(np.mean((preds - actuals) ** 2))

print("\n📊 RESULTS")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# -----------------------------
# PLOT 1: PRED vs ACTUAL
# -----------------------------

plt.figure(figsize=(6,6))
plt.scatter(actuals, preds)

plt.plot([0,100], [0,100])  # perfect line

plt.xlabel("Actual Defoliation (%)")
plt.ylabel("Predicted Defoliation (%)")
plt.title("Prediction vs Actual")

plt.grid()
plt.show()

# -----------------------------
# PLOT 2: ERROR DISTRIBUTION
# -----------------------------

errors = preds - actuals

plt.figure(figsize=(6,4))
plt.hist(errors, bins=20)

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")

plt.grid()
plt.show()

# -----------------------------
# PLOT 3: ERROR vs ACTUAL
# -----------------------------

plt.figure(figsize=(6,4))
plt.scatter(actuals, errors)

plt.axhline(0)

plt.xlabel("Actual Defoliation")
plt.ylabel("Error (Pred - Actual)")
plt.title("Error vs Actual")

plt.grid()
plt.show()