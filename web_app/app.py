from flask import Flask, render_template, send_file, request
import requests
import os

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------

# compute node API
COMPUTE_NODE_URL = "http://localhost:5000/predict"

# using my local test images for now
RECONSTRUCTION_BASE = r"D:\Video_model\outputs\test_images"

# change this to actual healthy images folder on server
HEALTHY_IMAGE_DIR = r"D:\GreenhouseDataset\reconstructed_healthy"


# -----------------------------
# IMAGE SERVING
# -----------------------------
@app.route("/image")
def serve_image():
    path = request.args.get("path")
    return send_file(path)


# -----------------------------
# MAIN DASHBOARD
# -----------------------------
@app.route("/")
def index():

    results = []

    # getting all images and sorting by latest first
    files = sorted(
        os.listdir(RECONSTRUCTION_BASE),
        key=lambda x: os.path.getmtime(os.path.join(RECONSTRUCTION_BASE, x)),
        reverse=True
    )

    for file_name in files:

        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        leaf_id = os.path.splitext(file_name)[0]
        reconstruction_img = os.path.join(RECONSTRUCTION_BASE, file_name)

        # sending image to compute node
        try:
            with open(reconstruction_img, "rb") as f:
                files_payload = {"image": f}
                response = requests.post(COMPUTE_NODE_URL, files=files_payload)
                data = response.json()

            matches = data.get("matches", [])
            print("matches for", leaf_id, ":", matches)

        except Exception as e:
            print("error:", e)
            matches = []

        # building paths for matched healthy images
        healthy_images = []

        for m in matches:
            # removing 'T' from simulated IDs to match healthy filenames
            clean_id = m.strip().replace("T", "", 1)
            img_name = clean_id + ".jpg"
            img_path = os.path.join(HEALTHY_IMAGE_DIR, img_name)

            if os.path.exists(img_path):
                healthy_images.append(img_path)
            else:
                print("missing:", img_path)
                healthy_images.append(reconstruction_img)  # fallback

        results.append({
            "leaf_id": leaf_id,
            "reconstruction": reconstruction_img,
            "matches": healthy_images
        })

    return render_template("index.html", results=results)


# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)