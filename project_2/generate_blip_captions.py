import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# ============================
# CONFIG
# ============================

MODEL_PATH = "blip_best_model.pth"          # your trained model
IMAGE_DIR = "val2017"                  # COCO val images
OUTPUT_JSON = "blip_val_captions.json"

BATCH_SIZE = 8                         # reduce if OOM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# LOAD MODEL
# ============================

print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded.")

# ============================
# LOAD IMAGE LIST
# ============================

image_files = sorted(os.listdir(IMAGE_DIR))

print("Total images:", len(image_files))

# ============================
# GENERATE CAPTIONS
# ============================

results = []

for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Generating captions"):

    batch_files = image_files[i:i+BATCH_SIZE]

    images = []
    image_ids = []

    for file in batch_files:

        path = os.path.join(IMAGE_DIR, file)

        image = Image.open(path).convert("RGB")
        images.append(image)

        image_id = int(file.split(".")[0])
        image_ids.append(image_id)

    inputs = processor(images=images, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_length=20)

    captions = processor.batch_decode(output, skip_special_tokens=True)

    for img_id, cap in zip(image_ids, captions):

        results.append({
            "image_id": img_id,
            "caption": cap.strip()
        })

# ============================
# SAVE JSON
# ============================

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)

print("Saved captions to:", OUTPUT_JSON)
print("Total captions:", len(results))