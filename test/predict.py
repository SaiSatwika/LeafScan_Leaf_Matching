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

print("\nStatus Code:", response.status_code)

try:
    data = response.json()

    print("\n Prediction:", data.get("prediction"), "%")
    print(" Confidence:", data.get("confidence"), "%")

    matches = data.get("matches", [])

    print("\n Top 5 Matches:")
    for i, m in enumerate(matches, 1):
        print(f"   {i}. {m}")

except:
    print("\nRaw Response:", response.text)