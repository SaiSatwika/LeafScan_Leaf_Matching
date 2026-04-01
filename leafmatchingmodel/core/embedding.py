import torch
from torchvision import models, transforms
from PIL import Image
import cv2


class EmbeddingModel:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.model.classifier = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def get_embedding(self, image):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)

        return emb.squeeze(0).cpu().numpy()