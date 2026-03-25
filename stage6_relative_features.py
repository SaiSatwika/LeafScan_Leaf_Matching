import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path.cwd()

GEOM_PATH = PROJECT_ROOT / "outputs/features/phase1_geometry.csv"
STAGE5_PATH = PROJECT_ROOT / "outputs/features/stage5_output.csv"

OUT_PATH = PROJECT_ROOT / "outputs/features/stage6_output.csv"

# -------------------------------------------------
# LOAD
# -------------------------------------------------

df_geom = pd.read_csv(GEOM_PATH)
df_stage5 = pd.read_csv(STAGE5_PATH)

# -------------------------------------------------
# FILTER SIMULATED
# -------------------------------------------------

df_sim = df_geom[df_geom["category"] == "simulated"].copy()

print("Simulated samples:", len(df_sim))
print("Stage5 rows:", len(df_stage5))

# -------------------------------------------------
# CLEAN IDS
# -------------------------------------------------

def normalize_id(name):
    return "_".join(str(name).strip().split())

df_sim["leaf_id"] = df_sim["leaf_id"].apply(normalize_id)
df_stage5["leaf_id"] = df_stage5["leaf_id"].apply(normalize_id)

# -------------------------------------------------
# MERGE
# -------------------------------------------------

df = df_sim.merge(df_stage5, on="leaf_id", how="inner")

print("After merge:", len(df))

# -------------------------------------------------
# FEATURE ENGINEERING (ORIGINAL — KEEP THIS)
# -------------------------------------------------

eps = 1e-6

df["rel_area_loss"] = (
    df["healthy_area_mean"] - df["area"]
) / (df["healthy_area_mean"] + eps)

df["rel_perimeter_change"] = (
    df["perimeter"] - df["healthy_perimeter_mean"]
) / (df["healthy_perimeter_mean"] + eps)

df["compactness_dev"] = (
    df["healthy_compactness_mean"] - df["compactness"]
)

df["convexity_dev"] = (
    df["healthy_convexity_mean"] - df["convexity"]
)

# clipping (important for stability)
df["rel_area_loss"] = df["rel_area_loss"].clip(0, 1.5)
df["rel_perimeter_change"] = df["rel_perimeter_change"].clip(-1.0, 1.5)

# -------------------------------------------------
# SAVE
# -------------------------------------------------

df.to_csv(OUT_PATH, index=False)

print("\nStage 6 complete")
print("Final dataset size:", len(df))