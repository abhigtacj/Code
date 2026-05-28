import os
import json
import random
import shutil
from tqdm import tqdm

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
COCO_IMG_DIR = "train2017"
COCO_ANN_FILE = "annotations/captions_train2017.json"

OUTPUT_IMG_DIR = "coco_subset/train2017_15k"
OUTPUT_ANN_FILE = "coco_subset/annotations/captions_train2017_15k.json"

NUM_IMAGES = 15000
RANDOM_SEED = 42
# --------------------------------------------------

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_ANN_FILE), exist_ok=True)

print("Loading COCO annotations...")
with open(COCO_ANN_FILE, "r") as f:
    coco_data = json.load(f)

images = coco_data["images"]
annotations = coco_data["annotations"]

print(f"Total images in dataset: {len(images)}")

# --------------------------------------------------
# Step 1: Randomly select 15k images
# --------------------------------------------------
random.seed(RANDOM_SEED)
selected_images = random.sample(images, NUM_IMAGES)

selected_image_ids = set(img["id"] for img in selected_images)

print(f"Selected {len(selected_images)} random images.")

# --------------------------------------------------
# Step 2: Filter annotations belonging to selected images
# --------------------------------------------------
filtered_annotations = [
    ann for ann in annotations
    if ann["image_id"] in selected_image_ids
]

print(f"Filtered annotations: {len(filtered_annotations)}")

# --------------------------------------------------
# Step 3: Create new annotation JSON
# --------------------------------------------------
filtered_coco = {
    "info": coco_data.get("info", {}),
    "licenses": coco_data.get("licenses", []),
    "images": selected_images,
    "annotations": filtered_annotations
}

with open(OUTPUT_ANN_FILE, "w") as f:
    json.dump(filtered_coco, f)

print("Saved filtered annotation file.")

# --------------------------------------------------
# Step 4: Copy selected images (optional but recommended)
# --------------------------------------------------
print("Copying image files...")

for img in tqdm(selected_images):
    src = os.path.join(COCO_IMG_DIR, img["file_name"])
    dst = os.path.join(OUTPUT_IMG_DIR, img["file_name"])
    shutil.copyfile(src, dst)

print("Done! 15k subset created successfully.")