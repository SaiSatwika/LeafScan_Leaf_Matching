import sys
from pathlib import Path

# -----------------------------
# FIX IMPORT PATH
# -----------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

import requests
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import re
import pickle

from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_distances

from matching_model.leaf_matching.utils.preprocess import extract_leaf
from matching_model.leaf_matching.utils.geometry import compute_geometry

# -----------------------------
# PATHS
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEALTHY_IMG_DIR = Path(r"D:\Final_dataset\Healthy")

HEALTHY_EMB_PATH = PROJECT_ROOT / "outputs/embeddings/healthy_embeddings.npy"
HEALTHY_ID_PATH = PROJECT_ROOT / "outputs/embeddings/healthy_leaf_ids.txt"
GEOM_PATH = PROJECT_ROOT / "outputs/features/phase1_geometry.csv"
GT_PATH = PROJECT_ROOT / "outputs/features/gt_mapping.csv"

# 🔥 CACHE FILE
CACHE_PATH = PROJECT_ROOT / "outputs/image_map.pkl"

# -----------------------------
# LOAD DATA
# -----------------------------

healthy_embeddings = np.load(HEALTHY_EMB_PATH)

with open(HEALTHY_ID_PATH) as f:
    healthy_ids = [l.strip() for l in f]

df_geom = pd.read_csv(GEOM_PATH)
df_gt = pd.read_csv(GT_PATH)

# -----------------------------
# NORMALIZATION
# -----------------------------

def normalize_id(name):
    return "_".join(str(name).strip().split()).lower()

df_geom["leaf_id"] = df_geom["leaf_id"].apply(normalize_id)
healthy_ids = [normalize_id(x) for x in healthy_ids]

df_healthy = (
    df_geom[df_geom["category"] == "healthy"]
    .set_index("leaf_id")
    .loc[healthy_ids]
    .reset_index()
)

# -----------------------------
# BUILD IMAGE MAP (🔥 CACHED)
# -----------------------------

def build_image_map(folder):

    # ✅ Load cache if exists
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "rb") as f:
            print("⚡ Loaded cached image map")
            return pickle.load(f)

    # ❌ Build map
    image_map = {}

    for path in folder.glob("*.jpg"):
        name = path.stem.lower()
        nums = "-".join(re.findall(r'\d+', name))

        if nums:
            image_map[nums] = str(path)

    # ✅ Save cache
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(image_map, f)

    print(f"✅ Built and cached image map ({len(image_map)} images)")

    return image_map

image_map = build_image_map(HEALTHY_IMG_DIR)

# -----------------------------
# GT HELPER
# -----------------------------

def extract_key_from_filename(filename):
    nums = re.findall(r'\d+', filename)

    key_full = "-".join(nums)

    if key_full in df_gt["leaf_key"].values:
        return key_full

    if len(nums) >= 5:
        key_trimmed = "-".join(nums[:-1])
        if key_trimmed in df_gt["leaf_key"].values:
            return key_trimmed

    return key_full

# -----------------------------
# EMBEDDING MODEL
# -----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
embed_model.classifier = torch.nn.Identity()
embed_model = embed_model.to(device)
embed_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = embed_model(tensor)
    return emb.squeeze(0).cpu().numpy()

# -----------------------------
# MATCHING
# -----------------------------

def match_healthy(sim_emb, geom, K=5):

    sim_length = geom["major_axis"]
    sim_width = geom["minor_axis"]
    sim_ar = sim_length / (sim_width + 1e-6)

    tolerance_levels = [0.2, 0.35, 0.5, 0.8]
    candidate_indices = None

    for tol in tolerance_levels:

        mask = (
            (df_healthy["major_axis"].between(sim_length*(1-tol), sim_length*(1+tol))) &
            (df_healthy["minor_axis"].between(sim_width*(1-tol), sim_width*(1+tol))) &
            (df_healthy["aspect_ratio"].between(sim_ar*(1-tol), sim_ar*(1+tol)))
        )

        idx = np.where(mask)[0]

        if len(idx) >= K:
            candidate_indices = idx
            break

    if candidate_indices is None:
        candidate_indices = np.arange(len(df_healthy))

    candidate_embeddings = healthy_embeddings[candidate_indices]

    dists = cosine_distances(
        sim_emb.reshape(1, -1),
        candidate_embeddings
    ).flatten()

    local_nn = np.argsort(dists)[:K]
    final_indices = candidate_indices[local_nn]

    return final_indices, dists[local_nn], df_healthy.iloc[final_indices]

# -----------------------------
# ARGUMENT
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True)
args = parser.parse_args()

# -----------------------------
# API CALL
# -----------------------------

url = "http://127.0.0.1:5000/predict"

with open(args.image, "rb") as f:
    response = requests.post(url, files={"image": f})

data = response.json()

pred = data.get("prediction")
confidence = data.get("confidence")

print("\nPrediction:", pred)
print("Confidence:", confidence)

# -----------------------------
# LOAD IMAGE
# -----------------------------

image = cv2.imread(args.image)
sim_leaf_id = Path(args.image).stem

# -----------------------------
# GT LOOKUP
# -----------------------------

leaf_key = extract_key_from_filename(sim_leaf_id)
gt_row = df_gt[df_gt["leaf_key"] == leaf_key]

actual = None
if len(gt_row) > 0:
    actual = float(gt_row["defoliation_gt"].values[0])

if actual is not None:
    print("Actual:", actual)
    print("Error:", round(abs(pred - actual), 2))
else:
    print("⚠️ GT not found for:", sim_leaf_id)

# -----------------------------
# LOCAL PIPELINE
# -----------------------------

mask, leaf = extract_leaf(image)
geom = compute_geometry(mask)
emb = get_embedding(leaf)

idx, dists, matched = match_healthy(emb, geom)

# -----------------------------
# VISUALIZATION
# -----------------------------

plt.figure(figsize=(20, 6))

# SIMULATED
plt.subplot(1, 6, 1)
plt.imshow(cv2.cvtColor(leaf, cv2.COLOR_BGR2RGB))
plt.title(
    f"SIM\n{sim_leaf_id}\n"
    f"L={geom['major_axis']:.1f} W={geom['minor_axis']:.1f}\n"
    f"AR={geom['aspect_ratio']:.2f}"
)
plt.axis("off")

# MATCHES
for i, (_, row) in enumerate(matched.iterrows()):

    nums = "-".join(re.findall(r'\d+', row["leaf_id"]))
    img_path = image_map.get(nums)

    if img_path is not None:
        img = cv2.imread(img_path)
    else:
        print("❌ Not found:", row["leaf_id"], "→", nums)
        img = np.zeros((200, 100, 3), dtype=np.uint8)

    plt.subplot(1, 6, i+2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(
        f"H{i+1}\n{row['leaf_id']}\n"
        f"L={row['major_axis']:.1f} W={row['minor_axis']:.1f}\n"
        f"AR={row['aspect_ratio']:.2f}\n"
        f"d={dists[i]:.3f}"
    )
    plt.axis("off")

# TITLE
title = f"Prediction: {pred}% | Confidence: {confidence}%"

if actual is not None:
    title += f" | Actual: {actual}% | Error: {abs(pred - actual):.2f}%"

plt.suptitle(title, fontsize=16)
plt.tight_layout()
plt.show()