import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# -------------------------------------------------
# PATHS
# -------------------------------------------------

PROJECT_ROOT = Path.cwd()

DATA_PATH = PROJECT_ROOT / "outputs/features/stage6_with_gt.csv"

HEALTHY_DIR = Path(r"D:\updated dataset\Healthy_reconstruction")
DEFOLIATED_DIR = Path(r"D:\updated dataset\Defoliated_reconstruction")

OUT_TEST_DIR = PROJECT_ROOT / "outputs/test_images"
OUT_TRAIN_DIR = PROJECT_ROOT / "outputs/train_images"

OUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
OUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# LOAD
# -------------------------------------------------

df = pd.read_csv(DATA_PATH)

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

print("Train:", len(train_df))
print("Test:", len(test_df))

# -------------------------------------------------
# KEY FIX FUNCTION
# -------------------------------------------------

def convert_leafid_to_filename(leaf_id):
    """
    Convert:
    1-1-9___D2 → 1-1-9 _ D2.jpg
    """
    if "___" in leaf_id:
        parts = leaf_id.split("___")
        return f"{parts[0]} _ {parts[1]}.jpg"
    else:
        return f"{leaf_id}.jpg"

# -------------------------------------------------
# COPY FUNCTION
# -------------------------------------------------

def copy_image(leaf_id, dest_folder):

    filename = convert_leafid_to_filename(leaf_id)

    path = DEFOLIATED_DIR / filename
    if path.exists():
        shutil.copy(path, dest_folder / filename)
        return True

    path = HEALTHY_DIR / filename
    if path.exists():
        shutil.copy(path, dest_folder / filename)
        return True

    print("Missing file:", filename)
    return False

# -------------------------------------------------
# COPY TEST
# -------------------------------------------------

missing = 0

for leaf_id in test_df["leaf_id"]:
    ok = copy_image(leaf_id, OUT_TEST_DIR)
    if not ok:
        missing += 1

print("\nTest images copied:", len(test_df) - missing)
print("Missing:", missing)

# -------------------------------------------------
# COPY TRAIN
# -------------------------------------------------

missing = 0

for leaf_id in train_df["leaf_id"]:
    ok = copy_image(leaf_id, OUT_TRAIN_DIR)
    if not ok:
        missing += 1

print("\nTrain images copied:", len(train_df) - missing)
print("Missing:", missing)