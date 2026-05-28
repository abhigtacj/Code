import json
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download('punkt')

# -----------------------------
# LOAD FILES
# -----------------------------
with open("outputs/blip_val_attention_captions.json", "r") as f:
    predictions = json.load(f)

with open("annotations/captions_val2017.json", "r") as f:
    coco_caps = json.load(f)

# -----------------------------
# BUILD GT DICTIONARY
# image_id → list of reference captions
# -----------------------------
gt_dict = defaultdict(list)

for ann in coco_caps["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"]
    gt_dict[image_id].append(caption)

# -----------------------------
# BLEU CALCULATION
# -----------------------------
smooth = SmoothingFunction().method1

bleu1 = []
bleu2 = []
bleu3 = []
bleu4 = []

for pred in predictions:
    image_id = pred["image_id"]
    pred_caption = pred["caption"]

    if image_id not in gt_dict:
        continue

    references = [nltk.word_tokenize(ref.lower()) for ref in gt_dict[image_id]]
    hypothesis = nltk.word_tokenize(pred_caption.lower())

    bleu1.append(sentence_bleu(references, hypothesis, weights=(1,0,0,0), smoothing_function=smooth))
    bleu2.append(sentence_bleu(references, hypothesis, weights=(0.5,0.5,0,0), smoothing_function=smooth))
    bleu3.append(sentence_bleu(references, hypothesis, weights=(0.33,0.33,0.33,0), smoothing_function=smooth))
    bleu4.append(sentence_bleu(references, hypothesis, weights=(0.25,0.25,0.25,0.25), smoothing_function=smooth))

print("BLEU-1:", sum(bleu1)/len(bleu1))
print("BLEU-2:", sum(bleu2)/len(bleu2))
print("BLEU-3:", sum(bleu3)/len(bleu3))
print("BLEU-4:", sum(bleu4)/len(bleu4))
