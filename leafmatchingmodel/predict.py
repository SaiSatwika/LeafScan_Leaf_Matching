import cv2
import numpy as np
from .model import LeafMatchingModel


_model_instance = None


def load_model(data_url):
    global _model_instance
    _model_instance = LeafMatchingModel(data_url)


def predict_from_path(image_path):

    if _model_instance is None:
        raise RuntimeError("Model not loaded")

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Invalid image")

    return _model_instance.predict(image)


def predict_from_bytes(image_bytes):

    if _model_instance is None:
        raise RuntimeError("Model not loaded")

    arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image")

    return _model_instance.predict(image)