import numpy as np


def compute_features(geom, matched, dists):

    h_area = float(matched["area"].mean())
    h_per = float(matched["perimeter"].mean())
    h_comp = float(matched["compactness"].mean())
    h_conv = float(matched["convexity"].mean())

    rel_area_loss = (h_area - geom["area"]) / (h_area + 1e-6)
    rel_perimeter_change = (geom["perimeter"] - h_per) / (h_per + 1e-6)
    compactness_dev = h_comp - geom["compactness"]
    convexity_dev = h_conv - geom["convexity"]

    features = np.array([[rel_area_loss,
                          convexity_dev,
                          float(dists.mean()),
                          rel_perimeter_change,
                          compactness_dev]])

    return features


def predict_defoliation(model, features, dists):

    pred = float(np.clip(model.predict(features)[0], 0, 100))

    confidence = float(max(0, 100 * (1 - float(dists.mean()))))
    confidence = float(round(confidence, 2))

    return round(pred, 2), confidence