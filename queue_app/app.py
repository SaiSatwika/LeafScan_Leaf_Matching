from flask import Flask, render_template, request, send_file
import os
import shutil

app = Flask(__name__)

# -----------------------------
# CONFIG
# -----------------------------

# incoming images come here
LANDING_QUEUE = r"PATH_TO_LANDING_QUEUE"

# move accepted images here
UPDATE_QUEUE = r"PATH_TO_UPDATE_QUEUE"

# move rejected images here
DISCARD_QUEUE = r"PATH_TO_DISCARD_QUEUE"


# -----------------------------
# IMAGE SERVING
# -----------------------------
@app.route("/image")
def serve_image():
    path = request.args.get("path")
    return send_file(path)


# -----------------------------
# MAIN PAGE
# -----------------------------
@app.route("/")
def queue():

    # if path is not set yet, just show message instead of crashing
    if not os.path.exists(LANDING_QUEUE):
        return "landing queue path not set yet"

    images = []

    # reading all images from landing queue
    for file_name in os.listdir(LANDING_QUEUE):

        if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(LANDING_QUEUE, file_name)

        images.append({
            "name": file_name,
            "path": img_path
        })

    return render_template("queue.html", images=images)


# -----------------------------
# KEEP
# -----------------------------
@app.route("/keep", methods=["POST"])
def keep():

    file_name = request.form.get("file")

    src = os.path.join(LANDING_QUEUE, file_name)
    dst = os.path.join(UPDATE_QUEUE, file_name)

    try:
        shutil.move(src, dst)
    except Exception as e:
        print("error moving file:", e)

    return {"status": "ok"}


# -----------------------------
# DISCARD
# -----------------------------
@app.route("/discard", methods=["POST"])
def discard():

    file_name = request.form.get("file")

    src = os.path.join(LANDING_QUEUE, file_name)
    dst = os.path.join(DISCARD_QUEUE, file_name)

    try:
        shutil.move(src, dst)
    except Exception as e:
        print("error moving file:", e)

    return {"status": "ok"}


# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)