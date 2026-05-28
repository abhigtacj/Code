import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
import os
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------

MODEL_PATH = "blip_attention_best.pth"

VAL_IMAGE_DIR = "val2017"
VAL_ANNOTATION = "annotations/captions_val2017.json"

OUTPUT_JSON = "blip_val_attention_captions.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------
# LOAD MODEL
# ------------------------

print("Loading BLIP model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

model = model.to(DEVICE)

model.eval()

print("Model loaded")

# ------------------------
# LOAD VAL DATA
# ------------------------

with open(VAL_ANNOTATION) as f:
    coco = json.load(f)

images = coco["images"]

print("Total validation images:", len(images))

# ------------------------
# CAPTION GENERATION
# ------------------------

results = []

progress = tqdm(images, desc="Generating captions")

for img in progress:

    image_id = img["id"]
    filename = img["file_name"]

    image_path = os.path.join(VAL_IMAGE_DIR, filename)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_length=30,
            num_beams=3
        )

    caption = processor.decode(output[0], skip_special_tokens=True)

    results.append({
        "image_id": image_id,
        "caption": caption
    })

# ------------------------
# SAVE JSON
# ------------------------

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)

print("\nCaptions saved to:", OUTPUT_JSON)