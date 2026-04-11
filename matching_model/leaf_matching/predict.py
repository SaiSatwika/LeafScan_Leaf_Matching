import numpy as np

from leaf_matching.utils.preprocess import extract_leaf
from leaf_matching.utils.geometry import compute_geometry
from leaf_matching.models.efficientnet import get_embedding
from leaf_matching.match import match_healthy

def run_prediction(image, model, healthy_embeddings, df_geom):

    # -----------------------------
    # STEP 1: Extract Leaf
    # -----------------------------
    mask, leaf = extract_leaf(image)

    if mask is None:
        raise ValueError("Leaf extraction failed")

    # -----------------------------
    # STEP 2: Geometry
    # -----------------------------
    geom = compute_geometry(mask)

    if geom is None:
        raise ValueError("Geometry computation failed")

    # -----------------------------
    # STEP 3: Embedding
    # -----------------------------
    emb = get_embedding(leaf)

    # -----------------------------
    # STEP 4: Matching
    # -----------------------------
    idx, dists, matched = match_healthy(emb, healthy_embeddings, df_geom)

    # -----------------------------
    # STEP 5: Feature Engineering
    # -----------------------------
    h_area = float(matched["area"].mean())
    h_per = float(matched["perimeter"].mean())
    h_comp = float(matched["compactness"].mean())
    h_conv = float(matched["convexity"].mean())

    rel_area_loss = float((h_area - geom["area"]) / (h_area + 1e-6))
    rel_perimeter_change = float((geom["perimeter"] - h_per) / (h_per + 1e-6))
    compactness_dev = float(h_comp - geom["compactness"])
    convexity_dev = float(h_conv - geom["convexity"])

    features = np.array([[
        rel_area_loss,
        convexity_dev,
        float(dists.mean()),
        rel_perimeter_change,
        compactness_dev
    ]])

    # -----------------------------
    # STEP 6: Prediction
    # -----------------------------
    pred = float(model.predict(features)[0])

    # 🔥 SIGN FIX (your requirement)
    if geom["area"] > h_area:
        pred = -abs(pred)
    else:
        pred = abs(pred)

    pred = round(pred, 2)

    # -----------------------------
    # CONFIDENCE
    # -----------------------------
    confidence = float(max(0, 100 * (1 - float(dists.mean()))))
    confidence = round(confidence, 2)

    # -----------------------------
    # MATCH IDS
    # -----------------------------
    matched_ids = list(matched["leaf_id"].values)

    return pred, confidence, matched_ids