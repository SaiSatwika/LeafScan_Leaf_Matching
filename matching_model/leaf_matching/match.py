import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def match_healthy(sim_emb, healthy_embeddings, df_geom, K=5):

    # Filter healthy leaves
    df_healthy = df_geom[df_geom["category"] == "healthy"]

    # Compute cosine distances
    dists = cosine_distances(
        sim_emb.reshape(1, -1),
        healthy_embeddings
    ).flatten()

    # Get top-K closest matches
    idx = np.argsort(dists)[:K]

    return idx, dists[idx], df_healthy.iloc[idx]