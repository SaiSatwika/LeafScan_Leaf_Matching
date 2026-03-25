import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances

# -------------------------------------------------
# PATHS
# -------------------------------------------------

PROJECT_ROOT = Path.cwd()

GEOM_PATH = PROJECT_ROOT / "outputs/features/phase1_geometry.csv"
EMB_DIR = PROJECT_ROOT / "outputs/embeddings"

HEALTHY_EMB_PATH = EMB_DIR / "healthy_embeddings.npy"
HEALTHY_ID_PATH = EMB_DIR / "healthy_leaf_ids.txt"
SIM_EMB_PATH = EMB_DIR / "simulated_embeddings.npy"
SIM_ID_PATH = EMB_DIR / "simulated_leaf_ids.txt"

OUT_PATH = PROJECT_ROOT / "outputs/features/stage5_output.csv"

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df_geom = pd.read_csv(GEOM_PATH)

healthy_embeddings = np.load(HEALTHY_EMB_PATH)
sim_embeddings = np.load(SIM_EMB_PATH)

with open(HEALTHY_ID_PATH, "r") as f:
    healthy_leaf_ids = [line.strip() for line in f]

with open(SIM_ID_PATH, "r") as f:
    sim_leaf_ids = [line.strip() for line in f]

print("Healthy embeddings:", healthy_embeddings.shape)
print("Sim embeddings:", sim_embeddings.shape)

# -------------------------------------------------
# NORMALIZE IDS
# -------------------------------------------------

def normalize_id(name):
    return "_".join(str(name).strip().split())

df_geom["leaf_id"] = df_geom["leaf_id"].apply(normalize_id)

healthy_ids_clean = [normalize_id(lid.replace("_norm", "")) for lid in healthy_leaf_ids]
sim_ids_clean = [normalize_id(lid.replace("_norm", "")) for lid in sim_leaf_ids]

df_healthy = df_geom[df_geom["category"] == "healthy"].copy()
df_sim = df_geom[df_geom["category"] == "simulated"].copy()

df_healthy = (
    df_healthy
    .set_index("leaf_id")
    .loc[healthy_ids_clean]
    .reset_index()
)

df_sim = (
    df_sim
    .set_index("leaf_id")
    .loc[sim_ids_clean]
    .reset_index()
)

print("Aligned healthy:", len(df_healthy))
print("Aligned simulated:", len(df_sim))

# -------------------------------------------------
# MATCHING (STABLE BASELINE)
# -------------------------------------------------

K = 5
records = []

for i, row in df_sim.iterrows():

    leaf_id = row["leaf_id"]
    sim_emb = sim_embeddings[i]

    sim_length = row["major_axis"]
    sim_width = row["minor_axis"]
    sim_ar = sim_length / (sim_width + 1e-6)

    tolerance_levels = [0.2, 0.35, 0.5, 0.8]
    candidate_indices = None

    # -------- SHAPE FILTER ONLY --------
    for tol in tolerance_levels:

        lower_L = sim_length * (1 - tol)
        upper_L = sim_length * (1 + tol)

        lower_W = sim_width * (1 - tol)
        upper_W = sim_width * (1 + tol)

        lower_AR = sim_ar * (1 - tol)
        upper_AR = sim_ar * (1 + tol)

        mask = (
            (df_healthy["major_axis"] >= lower_L) &
            (df_healthy["major_axis"] <= upper_L) &
            (df_healthy["minor_axis"] >= lower_W) &
            (df_healthy["minor_axis"] <= upper_W) &
            (df_healthy["aspect_ratio"] >= lower_AR) &
            (df_healthy["aspect_ratio"] <= upper_AR)
        )

        idx = np.where(mask)[0]

        if len(idx) >= K:
            candidate_indices = idx
            break

    # fallback
    if candidate_indices is None or len(candidate_indices) < K:
        candidate_indices = np.arange(len(df_healthy))

    # -------- PURE EMBEDDING MATCH --------
    candidate_embeddings = healthy_embeddings[candidate_indices]

    dists = cosine_distances(
        sim_emb.reshape(1, -1),
        candidate_embeddings
    ).flatten()

    local_nn = np.argsort(dists)[:K]
    final_indices = candidate_indices[local_nn]

    matched = df_healthy.iloc[final_indices]

    records.append({
        "leaf_id": leaf_id,
        "healthy_area_mean": matched["area"].mean(),
        "healthy_perimeter_mean": matched["perimeter"].mean(),
        "healthy_compactness_mean": matched["compactness"].mean(),
        "healthy_convexity_mean": matched["convexity"].mean(),
        "mean_embedding_distance": dists[local_nn].mean()
    })

# -------------------------------------------------
# SAVE
# -------------------------------------------------

df_stage5 = pd.DataFrame(records)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_stage5.to_csv(OUT_PATH, index=False)

print("\nStage 5 complete (STABLE BASELINE)")
print("Rows:", len(df_stage5))