import pandas as pd
import re
from pathlib import Path

DATA_ROOT = Path(r"D:\updated dataset")

FILES = [
    "defoliated_index.csv",
    "tattered_defo_index.csv"
]

def extract_key_from_csv(human_id):
    """
    Converts:
    1-3 > Leaf 9 (D1)
    OR
    1-3-9 > D1

    → 1-3-9-1
    """

    nums = re.findall(r'\d+', str(human_id))
    return "-".join(nums)  # KEEP ALL numbers (important)

gt_map = {}

for file in FILES:
    path = DATA_ROOT / file

    if not path.exists():
        print(f"Missing file: {file}")
        continue

    df = pd.read_csv(path)

    print("\nProcessing:", file)
    print("Columns:", df.columns.tolist())

    for _, row in df.iterrows():

        if "defoliation" not in df.columns:
            continue

        defo = row["defoliation"]

        if defo == -1:
            continue

        key = extract_key_from_csv(row["human_readable_id"])

        gt_map[key] = defo

print("\nGT entries:", len(gt_map))

out_df = pd.DataFrame([
    {"leaf_key": k, "defoliation_gt": v}
    for k, v in gt_map.items()
])

out_path = Path("outputs/features/gt_mapping.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)

out_df.to_csv(out_path, index=False)

print("Saved:", out_path)