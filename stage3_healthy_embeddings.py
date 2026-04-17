import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

# -------------------------------------------------
# PATHS (UPDATED)
# -------------------------------------------------

HEALTHY_DIR = Path(r"D:\Final_dataset\Healthy")

OUTPUT_DIR = Path.cwd() / "outputs" / "embeddings"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# TRANSFORM
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# MODEL
# -------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = nn.Identity()
model = model.to(device)
model.eval()

# -------------------------------------------------
# EXTRACTION
# -------------------------------------------------

embeddings = []
leaf_ids = []

image_paths = sorted(list(HEALTHY_DIR.glob("*.*")))

print(f"Found {len(image_paths)} healthy+tattered images")

with torch.no_grad():
    for img_path in tqdm(image_paths):

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        emb = model(img_tensor)
        emb = emb.squeeze(0).cpu().numpy()

        embeddings.append(emb)
        leaf_ids.append(img_path.stem)

# -------------------------------------------------
# SAVE
# -------------------------------------------------

embeddings = np.stack(embeddings)

np.save(OUTPUT_DIR / "healthy_embeddings.npy", embeddings)

with open(OUTPUT_DIR / "healthy_leaf_ids.txt", "w") as f:
    for lid in leaf_ids:
        f.write(lid + "\n")

print("✅ Healthy (healthy+tattered) embeddings complete")
print("Shape:", embeddings.shape)