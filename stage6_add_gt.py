import pandas as pd
import re
from pathlib import Path

# -------------------------------------------------
# PATHS
# -------------------------------------------------

PROJECT_ROOT = Path.cwd()

STAGE6_PATH = PROJECT_ROOT / "outputs/features/stage6_output.csv"
GT_PATH = PROJECT_ROOT / "outputs/features/gt_mapping.csv"

OUT_PATH = PROJECT_ROOT / "outputs/features/stage6_with_gt.csv"

# -------------------------------------------------
# CHECK FILES
# -------------------------------------------------

if not STAGE6_PATH.exists():
    print(f"ERROR: Stage 6 file not found: {STAGE6_PATH}")
    exit()

if not GT_PATH.exists():
    print(f"ERROR: GT mapping file not found: {GT_PATH}")
    exit()

# -------------------------------------------------
# LOAD
# -------------------------------------------------

df = pd.read_csv(STAGE6_PATH)
df_gt = pd.read_csv(GT_PATH)

# normalize GT keys (remove 'd' if present)
df_gt["leaf_key"] = df_gt["leaf_key"].astype(str).str.lower()
df_gt["leaf_key"] = df_gt["leaf_key"].apply(lambda x: "-".join(re.findall(r'\d+', x)))

gt_keys = set(df_gt["leaf_key"])

# -------------------------------------------------
# SMART KEY EXTRACTION
# -------------------------------------------------

def extract_key_from_leaf_id(leaf_id):
    nums = re.findall(r'\d+', str(leaf_id))

    key_5 = "-".join(nums)

    # try exact match first
    if key_5 in gt_keys:
        return key_5

    # fallback to 4-term (drop scan)
    if len(nums) >= 5:
        key_4 = "-".join(nums[:-1])
        return key_4

    return key_5

df["leaf_key"] = df["leaf_id"].apply(extract_key_from_leaf_id)

# -------------------------------------------------
# DEBUG PRINT
# -------------------------------------------------

print("\nSample extracted keys:")
print(df[["leaf_id", "leaf_key"]].head())

print("\nSample GT keys:")
print(df_gt["leaf_key"].head())

# -------------------------------------------------
# MERGE
# -------------------------------------------------

df_merged = df.merge(df_gt, on="leaf_key", how="left")

# -------------------------------------------------
# STATS
# -------------------------------------------------

missing = df_merged["defoliation_gt"].isna().sum()

print("\nTotal rows:", len(df))
print("Missing GT:", missing)

# -------------------------------------------------
# CLEAN
# -------------------------------------------------

df_valid = df_merged.dropna(subset=["defoliation_gt"])

print("After cleaning:", len(df_valid))

# -------------------------------------------------
# GROUP BY LEAF (REMOVE SCAN NOISE)
# -------------------------------------------------

def get_leaf_base(key):
    parts = key.split("-")
    if len(parts) >= 5:
        return "-".join(parts[:-1])  # remove scan
    return key

df_valid["leaf_base"] = df_valid["leaf_key"].apply(get_leaf_base)

df_grouped = df_valid.groupby("leaf_base").agg({
    "rel_area_loss": "mean",
    "convexity_dev": "mean",
    "mean_embedding_distance": "mean",
    "rel_perimeter_change": "mean",
    "compactness_dev": "mean",
    "defoliation_gt": "mean"
}).reset_index()

print("After grouping (unique leaves):", len(df_grouped))

# -------------------------------------------------
# SAVE
# -------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_grouped.to_csv(OUT_PATH, index=False)

print("\nSaved:", OUT_PATH)