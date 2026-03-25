import requests
import argparse

# -----------------------------
# ARGUMENT PARSER
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str, required=True, help="Path to input image")
args = parser.parse_args()

# -----------------------------
# API CALL
# -----------------------------

url = "http://127.0.0.1:5000/predict"

with open(args.image, "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

# -----------------------------
# OUTPUT
# -----------------------------

print("Status Code:", response.status_code)

try:
    print("Response:", response.json())
except:
    print("Raw Response:", response.text)