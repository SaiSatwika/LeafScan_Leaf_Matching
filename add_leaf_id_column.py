import pandas as pd
from pathlib import Path

# -----------------------------
# PATH
# -----------------------------
DATA_DIR = Path(r"D:\GreenhouseDataset 4-6\Dataset")

INPUT_FILE = DATA_DIR / "defoliated_results.csv"
OUTPUT_FILE = DATA_DIR / "defoliated_results_updated.csv"

# -----------------------------
# LOAD
# -----------------------------
df = pd.read_csv(INPUT_FILE)

print("Columns:", df.columns.tolist())

# -----------------------------
# HELPERS
# -----------------------------

def format_entry(entry):
    # handles defo1 or defo_1 → d1
    entry = str(entry).lower()
    num = ''.join([c for c in entry if c.isdigit()])
    return f"d{num}"

# -----------------------------
# CREATE full_leaf_id
# -----------------------------

full_ids = []

for _, row in df.iterrows():
    try:
        base = str(row["base_leaf_id"]).strip()   # e.g. 1-1-16
        entry = format_entry(row["entry_type"])   # d1
        scan = str(int(row["scan_number"]))       # 1

        full_id = f"{base}-{entry}-{scan}"
        full_ids.append(full_id)

    except Exception as e:
        full_ids.append(None)

# insert as FIRST column
df.insert(0, "full_leaf_id", full_ids)

# -----------------------------
# SAVE
# -----------------------------
if OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:", OUTPUT_FILE)
print("Rows:", len(df))