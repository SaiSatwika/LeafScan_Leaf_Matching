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
    print("Run stage6_relative_features.py first.")
    exit()

if not GT_PATH.exists():
    print(f"ERROR: GT mapping file not found: {GT_PATH}")
    print("Run build_ground_truth.py first.")
    exit()

# -------------------------------------------------
# LOAD
# -------------------------------------------------

df = pd.read_csv(STAGE6_PATH)
df_gt = pd.read_csv(GT_PATH)

# -------------------------------------------------
# KEY EXTRACTION
# -------------------------------------------------

def extract_key_from_leaf_id(leaf_id):
    """
    Converts:
    1-1-9___D2 → 1-1-9-2
    1-3-10___D1 → 1-3-10-1
    """

    nums = re.findall(r'\d+', str(leaf_id))
    return "-".join(nums)

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

df_final = df_merged.dropna(subset=["defoliation_gt"])

print("After cleaning:", len(df_final))

# -------------------------------------------------
# SAVE
# -------------------------------------------------

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUT_PATH, index=False)

print("\nSaved:", OUT_PATH)