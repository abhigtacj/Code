import os
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# ======================
# CONFIG
# ======================
IMAGE_DIR = "val2017"
FEATURE_DIR = "features_val"
BATCH_SIZE = 64   # safe for 1650 Ti
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(FEATURE_DIR, exist_ok=True)

# ======================
# MODEL
# ======================
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(DEVICE)

# ======================
# TRANSFORM
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ======================
# FEATURE EXTRACTION
# ======================
images = sorted(os.listdir(IMAGE_DIR))

with torch.no_grad():
    for img_name in tqdm(images, desc="Extracting CNN features"):
        img_path = os.path.join(IMAGE_DIR, img_name)
        save_path = os.path.join(FEATURE_DIR, img_name.replace(".jpg", ".pt"))

        if os.path.exists(save_path):
            continue

        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        feature = resnet(image)
        feature = feature.squeeze().cpu()  # shape: (2048,)

        torch.save(feature, save_path)

print("Feature extraction complete.")
