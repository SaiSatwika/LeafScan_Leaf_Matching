import torch
from torchvision import models, transforms
from PIL import Image
import cv2

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MODEL
# -----------------------------
model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()
model = model.to(device)
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# EMBEDDING FUNCTION
# -----------------------------
def get_embedding(image):

    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model(tensor)

    return emb.squeeze(0).cpu().numpy()