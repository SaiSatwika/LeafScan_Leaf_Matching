import argparse
import cv2
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

from pathlib import Path
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_distances

from core.extract_leaf import extract_leaf

# -------------------------------------------------
# PATHS
# -------------------------------------------------

PROJECT_ROOT = Path.cwd()

MODEL_PATH = PROJECT_ROOT / "outputs/model/defoliation_model.pkl"
GEOM_PATH = PROJECT_ROOT / "outputs/features/phase1_geometry.csv"
GT_PATH = PROJECT_ROOT / "outputs/features/gt_mapping.csv"

HEALTHY_EMB_PATH = PROJECT_ROOT / "outputs/embeddings/healthy_embeddings.npy"
HEALTHY_ID_PATH = PROJECT_ROOT / "outputs/embeddings/healthy_leaf_ids.txt"

HEALTHY_IMG_DIR = Path(r"D:\updated dataset\Healthy_reconstruction")

# -------------------------------------------------
# LOAD
# -------------------------------------------------

model = joblib.load(MODEL_PATH)
healthy_embeddings = np.load(HEALTHY_EMB_PATH)

with open(HEALTHY_ID_PATH) as f:
    healthy_ids = [l.strip() for l in f]

df_geom = pd.read_csv(GEOM_PATH)
df_healthy = df_geom[df_geom["category"] == "healthy"].copy()

df_gt = pd.read_csv(GT_PATH)

# -------------------------------------------------
# UTILS
# -------------------------------------------------

def extract_key_from_filename(filename):
    nums = re.findall(r'\d+', filename)
    return "-".join(nums)

def compute_confidence(dists):
    mean_dist = np.mean(dists)
    confidence = max(0, 100 * (1 - mean_dist))
    return round(confidence, 2)

# -------------------------------------------------
# EMBEDDING MODEL
# -------------------------------------------------

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

# -------------------------------------------------
# GEOMETRY
# -------------------------------------------------

def compute_geometry(mask):

    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)

    convexity = area / hull_area if hull_area > 0 else 0.0
    compactness = area / (perimeter ** 2) if perimeter > 0 else 0.0

    pts = cnt.reshape(-1, 2).astype(np.float32)
    _, eigvecs, eigvals = cv2.PCACompute2(pts, mean=None)

    major_axis = 2 * np.sqrt(eigvals[0][0])
    minor_axis = 2 * np.sqrt(eigvals[1][0])
    aspect_ratio = major_axis / (minor_axis + 1e-6)

    return {
        "area": area,
        "perimeter": perimeter,
        "convexity": convexity,
        "compactness": compactness,
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "aspect_ratio": aspect_ratio
    }

# -------------------------------------------------
# MATCHING
# -------------------------------------------------

def match_healthy(sim_emb, geom, K=5):

    sim_length = geom["major_axis"]

    tolerance_levels = [0.15, 0.25, 0.35, 0.50]
    candidate_indices = None

    for tol in tolerance_levels:

        lower = sim_length * (1 - tol)
        upper = sim_length * (1 + tol)

        mask = (
            (df_healthy["major_axis"] >= lower) &
            (df_healthy["major_axis"] <= upper)
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

# -------------------------------------------------
# DEMO
# -------------------------------------------------

def run_demo(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print("Invalid image")
        return

    sim_leaf_id = Path(image_path).stem

    # ---- GT ----
    leaf_key = extract_key_from_filename(sim_leaf_id)
    gt_row = df_gt[df_gt["leaf_key"] == leaf_key]

    actual = None
    if len(gt_row) > 0:
        actual = float(gt_row["defoliation_gt"].values[0])

    # ---- PIPELINE ----
    mask, leaf = extract_leaf(image)
    geom = compute_geometry(mask)

    if geom is None:
        print("Geometry failed")
        return

    emb = get_embedding(leaf)

    idx, dists, matched = match_healthy(emb, geom)

    confidence = compute_confidence(dists)

    # ---- FEATURES ----
    h_area = matched["area"].mean()
    h_per = matched["perimeter"].mean()
    h_comp = matched["compactness"].mean()
    h_conv = matched["convexity"].mean()

    rel_area_loss = (h_area - geom["area"]) / (h_area + 1e-6)
    rel_perimeter_change = (geom["perimeter"] - h_per) / (h_per + 1e-6)
    compactness_dev = h_comp - geom["compactness"]
    convexity_dev = h_conv - geom["convexity"]

    features = np.array([[rel_area_loss,
                          convexity_dev,
                          dists.mean(),
                          rel_perimeter_change,
                          compactness_dev]])

    pred = float(np.clip(model.predict(features)[0], 0, 100))

    print("\nPredicted:", round(pred, 2))
    print("Actual:", actual if actual is not None else "Not found")
    print("Confidence:", confidence)

    # -------------------------------------------------
    # VISUALIZATION
    # -------------------------------------------------

    plt.figure(figsize=(20, 6))

    # SIM
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

        img_name = row["leaf_id"] + ".jpg"
        img_path = HEALTHY_IMG_DIR / img_name

        img = cv2.imread(str(img_path))

        if img is None:
            img = np.zeros((300, 100, 3), dtype=np.uint8)

        plt.subplot(1, 6, i + 2)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(
            f"H{i+1}\n{row['leaf_id']}\n"
            f"L={row['major_axis']:.1f} W={row['minor_axis']:.1f}\n"
            f"AR={row['aspect_ratio']:.2f}\n"
            f"d={dists[i]:.3f}"
        )
        plt.axis("off")

    # TITLE
    title = f"{sim_leaf_id} | Predicted: {pred:.2f}%"

    if actual is not None:
        title += f" | Actual: {actual:.2f}%"

    title += f" | Confidence: {confidence:.1f}%"

    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------
# CLI
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    run_demo(args.image)