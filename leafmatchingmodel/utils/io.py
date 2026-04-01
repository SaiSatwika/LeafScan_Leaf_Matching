import numpy as np
import pandas as pd
import joblib

from .paths import copy_resource_to_tmp


BASE_PACKAGE = "leafmatchingmodel.files"


def load_model():
    path = copy_resource_to_tmp(BASE_PACKAGE, "model.pkl")
    return joblib.load(path)


def load_embeddings():
    path = copy_resource_to_tmp(BASE_PACKAGE, "emb.npy")
    return np.load(path)


def load_ids():
    path = copy_resource_to_tmp(BASE_PACKAGE, "ids.txt")
    with open(path) as f:
        return [l.strip() for l in f]


def load_geom():
    path = copy_resource_to_tmp(BASE_PACKAGE, "geom.csv")
    return pd.read_csv(path)


def load_gt():
    path = copy_resource_to_tmp(BASE_PACKAGE, "gt.csv")
    return pd.read_csv(path)