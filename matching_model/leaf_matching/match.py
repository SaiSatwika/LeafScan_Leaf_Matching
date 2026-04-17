import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def normalize_id(name):
    return "_".join(str(name).strip().split())

def prepare_healthy_data(df_geom, healthy_embeddings, healthy_ids):

    df_geom["leaf_id"] = df_geom["leaf_id"].apply(normalize_id)
    healthy_ids = [normalize_id(x) for x in healthy_ids]

    df_healthy = df_geom[df_geom["category"] == "healthy"].copy()

    df_healthy = (
        df_healthy
        .set_index("leaf_id")
        .loc[healthy_ids]
        .reset_index()
    )

    return df_healthy, healthy_embeddings


def match_healthy(sim_emb, healthy_embeddings, df_geom, sim_geom, healthy_ids, K=5):

    df_healthy, healthy_embeddings = prepare_healthy_data(
        df_geom,
        healthy_embeddings,
        healthy_ids
    )

    sim_length = sim_geom["major_axis"]
    sim_width = sim_geom["minor_axis"]
    sim_ar = sim_length / (sim_width + 1e-6)

    tolerance_levels = [0.2, 0.35, 0.5, 0.8]
    candidate_indices = None

    for tol in tolerance_levels:

        lower_L = sim_length * (1 - tol)
        upper_L = sim_length * (1 + tol)

        lower_W = sim_width * (1 - tol)
        upper_W = sim_width * (1 + tol)

        lower_AR = sim_ar * (1 - tol)
        upper_AR = sim_ar * (1 + tol)

        mask = (
            (df_healthy["major_axis"] >= lower_L) &
            (df_healthy["major_axis"] <= upper_L) &
            (df_healthy["minor_axis"] >= lower_W) &
            (df_healthy["minor_axis"] <= upper_W) &
            (df_healthy["aspect_ratio"] >= lower_AR) &
            (df_healthy["aspect_ratio"] <= upper_AR)
        )

        idx = np.where(mask)[0]

        if len(idx) >= K:
            candidate_indices = idx
            break

    if candidate_indices is None or len(candidate_indices) < K:
        candidate_indices = np.arange(len(df_healthy))

    candidate_embeddings = healthy_embeddings[candidate_indices]

    dists = cosine_distances(
        sim_emb.reshape(1, -1),
        candidate_embeddings
    ).flatten()

    local_nn = np.argsort(dists)[:K]
    final_indices = candidate_indices[local_nn]

    matched = df_healthy.iloc[final_indices]

    return final_indices, dists[local_nn], matched