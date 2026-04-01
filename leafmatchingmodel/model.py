from .utils.io import (
    load_model,
    load_embeddings,
    load_ids,
    load_geom,
    load_gt
)

from .core.embedding import EmbeddingModel
from .core.geometry import compute_geometry
from .core.match import match_healthy
from .core.defoliation import compute_features, predict_defoliation
from .core.extract_leaf import extract_leaf


class LeafMatchingModel:

    def __init__(self):
        self.embedder = EmbeddingModel()
        self._load_all()

    def _load_all(self):
        self.model = load_model()
        self.healthy_embeddings = load_embeddings()
        self.healthy_ids = load_ids()
        self.df_geom = load_geom()
        self.df_gt = load_gt()

    def refresh(self):
        self._load_all()

    def predict(self, image):

        mask, leaf = extract_leaf(image)
        if mask is None:
            raise ValueError("Leaf extraction failed")

        geom = compute_geometry(mask)
        if geom is None:
            raise ValueError("Geometry failed")

        emb = self.embedder.get_embedding(leaf)

        idx, dists, matched = match_healthy(
            emb,
            self.healthy_embeddings,
            self.df_geom
        )

        features = compute_features(geom, matched, dists)

        pred, confidence = predict_defoliation(
            self.model,
            features,
            dists
        )

        return {
            "prediction": pred,
            "confidence": confidence
        }