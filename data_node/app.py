from flask import Flask, send_file, jsonify
from pathlib import Path

app = Flask(__name__)

# -----------------------------
# BASE PATH
# -----------------------------

BASE = Path(__file__).resolve().parent.parent / "outputs"

MODEL_PATH = BASE / "model" / "defoliation_model.pkl"
EMB_PATH = BASE / "embeddings" / "healthy_embeddings.npy"
ID_PATH = BASE / "embeddings" / "healthy_leaf_ids.txt"
GEOM_PATH = BASE / "features" / "phase1_geometry.csv"
GT_PATH = BASE / "features" / "gt_mapping.csv"

# -----------------------------
# FILE SERVING APIs
# -----------------------------

@app.route("/model")
def get_model():
    if not MODEL_PATH.exists():
        return jsonify({"error": "Model not found"}), 404
    return send_file(MODEL_PATH, as_attachment=True)

@app.route("/healthy-embeddings")
def get_embeddings():
    if not EMB_PATH.exists():
        return jsonify({"error": "Embeddings not found"}), 404
    return send_file(EMB_PATH, as_attachment=True)

@app.route("/healthy-ids")
def get_ids():
    if not ID_PATH.exists():
        return jsonify({"error": "IDs not found"}), 404
    return send_file(ID_PATH, as_attachment=True)

@app.route("/geometry")
def get_geometry():
    if not GEOM_PATH.exists():
        return jsonify({"error": "Geometry not found"}), 404
    return send_file(GEOM_PATH, as_attachment=True)

@app.route("/gt")
def get_gt():
    if not GT_PATH.exists():
        return jsonify({"error": "GT not found"}), 404
    return send_file(GT_PATH, as_attachment=True)

# -----------------------------
# HEALTH CHECK
# -----------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

# -----------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)