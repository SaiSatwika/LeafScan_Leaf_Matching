from flask import Flask, request, jsonify
import requests
import cv2
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import sys

# -----------------------------
# FIX IMPORT PATH (IMPORTANT)
# -----------------------------
sys.path.append(str(Path(__file__).resolve().parent.parent))

# UPDATED IMPORT
from matching_model.leaf_matching.predict import run_prediction

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
# LOAD DATA
# -----------------------------

def load_all():
    global model, healthy_embeddings, df_geom, healthy_ids

    print("⬇️ Downloading from Data Node...")

    download_file("model", CACHE_DIR / "model.pkl")
    download_file("healthy-embeddings", CACHE_DIR / "emb.npy")
    download_file("geometry", CACHE_DIR / "geom.csv")
    download_file("healthy-ids", CACHE_DIR / "healthy_leaf_ids.txt")  # ✅ NEW

    print("📦 Loading...")

    model = joblib.load(CACHE_DIR / "model.pkl")
    healthy_embeddings = np.load(CACHE_DIR / "emb.npy")
    df_geom = pd.read_csv(CACHE_DIR / "geom.csv")

    # ✅ LOAD IDS
    with open(CACHE_DIR / "healthy_leaf_ids.txt", "r") as f:
        healthy_ids = [line.strip() for line in f]

    print("✅ Ready")

# -----------------------------
# INITIAL LOAD
# -----------------------------

load_all()

# -----------------------------
# HEALTH
# -----------------------------

@app.route("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# PREDICT
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["image"]

        image_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # ✅ UPDATED CALL
        pred, confidence, matched_ids = run_prediction(
            image,
            model,
            healthy_embeddings,
            df_geom,
            healthy_ids   # ✅ IMPORTANT
        )

        return jsonify({
            "prediction": float(pred),
            "confidence": float(confidence),
            "matches": matched_ids
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# REFRESH
# -----------------------------

@app.route("/refresh", methods=["POST"])
def refresh():
    load_all()
    return {"status": "reloaded"}

# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)