import json
import nltk
from collections import defaultdict

#nltk.download('punkt')

# -----------------------------
# LOAD FILES
# -----------------------------
with open("outputs/blip_val_attention_captions.json", "r") as f:
    predictions = json.load(f)

with open("annotations/instances_val2017.json", "r") as f:
    coco_instances = json.load(f)

# -----------------------------
# COCO CATEGORY MAPPING
# category_id → name
# -----------------------------
cat_id_to_name = {}
for cat in coco_instances["categories"]:
    cat_id_to_name[cat["id"]] = cat["name"]

# -----------------------------
# image_id → set of true objects
# -----------------------------
image_objects = defaultdict(set)

for ann in coco_instances["annotations"]:
    image_id = ann["image_id"]
    category_id = ann["category_id"]
    obj_name = cat_id_to_name[category_id]
    image_objects[image_id].add(obj_name.lower())

# -----------------------------
# CHAIR CALCULATION
# -----------------------------
hallucinated_objects = 0
total_object_mentions = 0

for pred in predictions:
    image_id = pred["image_id"]
    caption = pred["caption"].lower()

    tokens = nltk.word_tokenize(caption)

    # Check each object word in caption
    for token in tokens:
        if token in cat_id_to_name.values():
            total_object_mentions += 1
            if token not in image_objects[image_id]:
                hallucinated_objects += 1

if total_object_mentions == 0:
    print("No object mentions found.")
else:
    chair = hallucinated_objects / total_object_mentions
    print("CHAIR Score:", chair)
    print("Hallucinated Objects:", hallucinated_objects)
    print("Total Object Mentions:", total_object_mentions)
