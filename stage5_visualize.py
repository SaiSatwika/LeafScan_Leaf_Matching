import cv2
import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances

from core.extract_leaf import extract_leaf
from normalize_leaf import normalize_leaf

# -------------------------------------------------
# PATHS
# -------------------------------------------------

HEALTHY_DIR = Path(r"D:\updated dataset\Healthy_reconstruction")
DEFOLIATED_DIR = Path(r"D:\updated dataset\Defoliated_reconstruction")

PROJECT_ROOT = Path.cwd()

GEOM_PATH = PROJECT_ROOT / "outputs/features/phase1_geometry.csv"
EMB_DIR = PROJECT_ROOT / "outputs/embeddings"

HEALTHY_EMB_PATH = EMB_DIR / "healthy_embeddings.npy"
HEALTHY_ID_PATH = EMB_DIR / "healthy_leaf_ids.txt"

SIM_EMB_PATH = EMB_DIR / "simulated_embeddings.npy"
SIM_ID_PATH = EMB_DIR / "simulated_leaf_ids.txt"

# -------------------------------------------------
# NORMALIZATION FOR MATCHING FILES
# -------------------------------------------------

def normalize_for_match(name):
    return "".join(
        name.lower()
        .replace(".jpg", "")
        .replace(".png", "")
        .replace(" ", "")
        .replace("_", "")
    )

# -------------------------------------------------
# ROBUST IMAGE FINDER
# -------------------------------------------------

def find_image(folder, leaf_id):

    target = normalize_for_match(leaf_id)

    for p in folder.glob("*.*"):
        fname = normalize_for_match(p.stem)

        if fname == target:
            return p

    return None

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df_geom = pd.read_csv(GEOM_PATH)

healthy_embeddings = np.load(HEALTHY_EMB_PATH)
sim_embeddings = np.load(SIM_EMB_PATH)

with open(HEALTHY_ID_PATH) as f:
    healthy_ids = [l.strip() for l in f]

with open(SIM_ID_PATH) as f:
    sim_ids = [l.strip() for l in f]

df_healthy = df_geom[df_geom["category"] == "healthy"].copy()
df_sim = df_geom[df_geom["category"] == "simulated"].copy()

df_healthy = df_healthy.set_index("leaf_id").loc[healthy_ids].reset_index()
df_sim = df_sim.set_index("leaf_id").loc[sim_ids].reset_index()

# -------------------------------------------------
# PICK RANDOM SIMULATED LEAF
# -------------------------------------------------

idx = random.randint(0, len(df_sim) - 1)

sim_row = df_sim.iloc[idx]
sim_id = sim_row["leaf_id"]
sim_emb = sim_embeddings[idx]

print(f"\n🔍 Visualizing for simulated leaf: {sim_id}")

# -------------------------------------------------
# MATCHING (same as Stage 5)
# -------------------------------------------------

K = 5

sim_length = sim_row["major_axis"]
sim_width = sim_row["minor_axis"]

tolerance_levels = [0.2, 0.35, 0.5, 0.8]

candidate_indices = None

for tol in tolerance_levels:

    lower_L = sim_length * (1 - tol)
    upper_L = sim_length * (1 + tol)

    lower_W = sim_width * (1 - tol)
    upper_W = sim_width * (1 + tol)

    mask = (
        (df_healthy["major_axis"] >= lower_L) &
        (df_healthy["major_axis"] <= upper_L) &
        (df_healthy["minor_axis"] >= lower_W) &
        (df_healthy["minor_axis"] <= upper_W)
    )

    idxs = np.where(mask)[0]

    if len(idxs) >= K:
        candidate_indices = idxs
        break

if candidate_indices is None or len(candidate_indices) < K:
    candidate_indices = np.arange(len(df_healthy))

candidate_embeddings = healthy_embeddings[candidate_indices]

dists = cosine_distances(
    sim_emb.reshape(1, -1),
    candidate_embeddings
).flatten()

topk_idx = np.argsort(dists)[:K]
final_indices = candidate_indices[topk_idx]

matched = df_healthy.iloc[final_indices]

# -------------------------------------------------
# LOAD SIMULATED IMAGE
# -------------------------------------------------

sim_path = find_image(DEFOLIATED_DIR, sim_id)

if sim_path is None:
    print(f"❌ Simulated image not found: {sim_id}")
    exit()

sim_img = cv2.imread(str(sim_path))

if sim_img is None:
    print(f"❌ Failed to read simulated image")
    exit()

sim_mask, sim_leaf = extract_leaf(sim_img)
sim_mask, sim_leaf = normalize_leaf(sim_mask, sim_leaf)

# -------------------------------------------------
# PLOT
# -------------------------------------------------

plt.figure(figsize=(20, 6))

# --- SIMULATED ---
plt.subplot(1, 6, 1)
plt.imshow(cv2.cvtColor(sim_leaf, cv2.COLOR_BGR2RGB))
plt.title(
    f"SIM\n{sim_id}\n"
    f"L={sim_row['major_axis']:.1f} W={sim_row['minor_axis']:.1f}\n"
    f"AR={sim_row['aspect_ratio']:.2f}"
)
plt.axis("off")

# --- MATCHES ---
for i, row in enumerate(matched.itertuples()):

    hid = row.leaf_id

    h_path = find_image(HEALTHY_DIR, hid)

    if h_path is None:
        print(f"⚠️ Missing healthy image: {hid}")
        continue

    h_img = cv2.imread(str(h_path))

    if h_img is None:
        print(f"⚠️ Failed to read: {h_path}")
        continue

    h_mask, h_leaf = extract_leaf(h_img)
    h_mask, h_leaf = normalize_leaf(h_mask, h_leaf)

    plt.subplot(1, 6, i + 2)
    plt.imshow(cv2.cvtColor(h_leaf, cv2.COLOR_BGR2RGB))
    plt.title(
        f"H{i+1}\n{hid}\n"
        f"L={row.major_axis:.1f} W={row.minor_axis:.1f}\n"
        f"AR={row.aspect_ratio:.2f}\n"
        f"d={dists[topk_idx[i]]:.3f}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()