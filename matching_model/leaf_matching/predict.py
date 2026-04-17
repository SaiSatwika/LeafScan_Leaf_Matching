import numpy as np

from matching_model.leaf_matching.utils.preprocess import extract_leaf
from matching_model.leaf_matching.utils.geometry import compute_geometry
from matching_model.leaf_matching.models.efficientnet import get_embedding
from matching_model.leaf_matching.match import match_healthy

def run_prediction(image, model, healthy_embeddings, df_geom, healthy_ids):

    mask, leaf = extract_leaf(image)
    if mask is None:
        raise ValueError("Leaf extraction failed")

    geom = compute_geometry(mask)
    if geom is None:
        raise ValueError("Geometry computation failed")

    emb = get_embedding(leaf)

    idx, dists, matched = match_healthy(
        emb,
        healthy_embeddings,
        df_geom,
        geom,
        healthy_ids
    )

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

    pred = float(model.predict(features)[0])

    if geom["area"] > h_area:
        pred = -abs(pred)
    else:
        pred = abs(pred)

    pred = round(pred, 2)

    confidence = float(max(0, 100 * (1 - float(dists.mean()))))
    confidence = round(confidence, 2)

    matched_ids = list(matched["leaf_id"].values)

    return pred, confidence, matched_ids