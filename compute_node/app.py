# -----------------------------
# FIX IMPORT PATH
# -----------------------------
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# -----------------------------
# IMPORTS
# -----------------------------
from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import pandas as pd
import joblib

from core.extract_leaf import extract_leaf

# -----------------------------
# CONFIG
# -----------------------------

DATA_NODE_URL = "http://localhost:5001"

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

# -----------------------------
# DOWNLOAD HELPER
# -----------------------------

def download_file(endpoint, save_path):
    url = f"{DATA_NODE_URL}/{endpoint}"
    r = requests.get(url)

    if r.status_code != 200:
        raise Exception(f"Failed to download {endpoint}")

    with open(save_path, "wb") as f:
        f.write(r.content)

# -----------------------------
# LOAD ALL FILES
# -----------------------------

def load_all():

    global model, healthy_embeddings, healthy_ids, df_geom, df_gt

    print("⬇️ Downloading files from Data Node...")

    download_file("model", CACHE_DIR / "model.pkl")
    download_file("healthy-embeddings", CACHE_DIR / "emb.npy")
    download_file("healthy-ids", CACHE_DIR / "ids.txt")
    download_file("geometry", CACHE_DIR / "geom.csv")
    download_file("gt", CACHE_DIR / "gt.csv")

    print("📦 Loading into memory...")

    model = joblib.load(CACHE_DIR / "model.pkl")
    healthy_embeddings = np.load(CACHE_DIR / "emb.npy")

    with open(CACHE_DIR / "ids.txt") as f:
        healthy_ids = [l.strip() for l in f]

    df_geom = pd.read_csv(CACHE_DIR / "geom.csv")
    df_gt = pd.read_csv(CACHE_DIR / "gt.csv")

    print("✅ Model + data loaded successfully")

# -----------------------------
# INITIAL LOAD
# -----------------------------

load_all()

# -----------------------------
# EMBEDDING MODEL
# -----------------------------

import torch
from torchvision import models, transforms
from PIL import Image

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
# GEOMETRY
# -----------------------------

def compute_geometry(mask):

    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)

    area = float(cv2.contourArea(cnt))
    perimeter = float(cv2.arcLength(cnt, True))

    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))

    convexity = float(area / hull_area) if hull_area > 0 else 0.0
    compactness = float(area / (perimeter ** 2)) if perimeter > 0 else 0.0

    pts = cnt.reshape(-1, 2).astype(np.float32)
    _, eigvecs, eigvals = cv2.PCACompute2(pts, mean=None)

    major_axis = float(2 * np.sqrt(eigvals[0][0]))
    minor_axis = float(2 * np.sqrt(eigvals[1][0]))

    return {
        "area": area,
        "perimeter": perimeter,
        "convexity": convexity,
        "compactness": compactness,
        "major_axis": major_axis,
        "minor_axis": minor_axis
    }

# -----------------------------
# MATCHING
# -----------------------------

from sklearn.metrics.pairwise import cosine_distances

def match_healthy(sim_emb, geom, K=5):

    df_healthy = df_geom[df_geom["category"] == "healthy"]

    dists = cosine_distances(
        sim_emb.reshape(1, -1),
        healthy_embeddings
    ).flatten()

    idx = np.argsort(dists)[:K]

    return idx, dists[idx], df_healthy.iloc[idx]

# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# -----------------------------
# PREDICT API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        file = request.files["image"]

        image_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image"}), 400

        mask, leaf = extract_leaf(image)

        if mask is None:
            return jsonify({"error": "Leaf extraction failed"}), 400

        geom = compute_geometry(mask)

        if geom is None:
            return jsonify({"error": "Geometry failed"}), 400

        emb = get_embedding(leaf)

        idx, dists, matched = match_healthy(emb, geom)

        # ---- FEATURES ----
        h_area = float(matched["area"].mean())
        h_per = float(matched["perimeter"].mean())
        h_comp = float(matched["compactness"].mean())
        h_conv = float(matched["convexity"].mean())

        rel_area_loss = float((h_area - geom["area"]) / (h_area + 1e-6))
        rel_perimeter_change = float((geom["perimeter"] - h_per) / (h_per + 1e-6))
        compactness_dev = float(h_comp - geom["compactness"])
        convexity_dev = float(h_conv - geom["convexity"])

        features = np.array([[rel_area_loss,
                              convexity_dev,
                              float(dists.mean()),
                              rel_perimeter_change,
                              compactness_dev]])

        pred = float(np.clip(model.predict(features)[0], 0, 100))

        # ✅ FIXED confidence (no numpy type)
        confidence = float(max(0, 100 * (1 - float(dists.mean()))))
        confidence = float(round(confidence, 2))

        return jsonify({
            "prediction": float(round(pred, 2)),
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# REFRESH
# -----------------------------

@app.route("/refresh", methods=["POST"])
def refresh():
    load_all()
    return jsonify({"status": "reloaded"})

# -----------------------------

if __name__ == "__main__":
    app.run(port=5000)