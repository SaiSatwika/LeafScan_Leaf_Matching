import pandas as pd
import re
from pathlib import Path

# -----------------------------
# PATH
# -----------------------------
GT_PATH = Path("outputs/features/gt_mapping.csv")

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(GT_PATH)

print("Before:", len(df))

# -----------------------------
# FIX ONLY ROWS 67–162
# (index 66 to 161)
# -----------------------------
for i in range(66, 161):

    key = str(df.loc[i, "leaf_key"]).lower()

    # remove 'd' (like d1 → 1)
    nums = re.findall(r'\d+', key)

    df.loc[i, "leaf_key"] = "-".join(nums)

# -----------------------------
# SAVE
# -----------------------------
df.to_csv(GT_PATH, index=False)

print("GT cleaned (removed 'd' from new rows)")