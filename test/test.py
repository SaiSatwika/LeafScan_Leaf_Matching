import cv2
from leafmatchingmodel import LeafMatchingModel

# Load model
model = LeafMatchingModel()

# Read image (BGR format)
image = cv2.imread("test/reconstruction.jpg")

if image is None:
    raise ValueError("Failed to load image")

# Run prediction
result = model.predict(image)

print(result)