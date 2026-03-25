import numpy as np
from sklearn.metrics.pairwise import cosine_distances

emb = np.load("outputs/embeddings/simulated_embeddings.npy")

print("Shape:", emb.shape)
print("Contains NaN:", np.isnan(emb).any())
print("Min:", emb.min())
print("Max:", emb.max())

# Diversity check
d = cosine_distances(emb[:5], emb[:5])
print("\nDistance matrix:\n", d)