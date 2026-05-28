import json
import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from pycocotools.coco import COCO

coco = COCO("annotations/captions_val2017.json")

gt_dict = {}

for img_id in coco.imgs.keys():
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    gt_dict[img_id] = [ann["caption"] for ann in anns]

# -------------------------------------------------
# Load generated captions
# -------------------------------------------------

with open("outputs/blip_val_attention_captions.json", "r") as f:
    predictions = json.load(f)

# -------------------------------------------------
# Load ground truth captions
# (Assuming you already built gt_dict earlier)
# If not, I include full version below.
# -------------------------------------------------

# gt_dict format:
# {
#   image_id1: ["caption1", "caption2", ...],
#   image_id2: ["caption1", "caption2", ...],
# }

# -------------------------------------------------
# Prepare data for corpus_bleu
# -------------------------------------------------

all_references = []
all_hypotheses = []

for pred in predictions:
    image_id = pred["image_id"]
    pred_caption = pred["caption"]

    if image_id not in gt_dict:
        continue

    # tokenize references
    references = [
        word_tokenize(ref.lower())
        for ref in gt_dict[image_id]
    ]

    # tokenize prediction
    hypothesis = word_tokenize(pred_caption.lower())

    all_references.append(references)
    all_hypotheses.append(hypothesis)

# -------------------------------------------------
# Compute Corpus BLEU
# -------------------------------------------------

bleu1 = corpus_bleu(all_references, all_hypotheses, weights=(1, 0, 0, 0))
bleu2 = corpus_bleu(all_references, all_hypotheses, weights=(0.5, 0.5, 0, 0))
bleu3 = corpus_bleu(all_references, all_hypotheses, weights=(0.33, 0.33, 0.33, 0))
bleu4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

print("Corpus BLEU-1:", bleu1)
print("Corpus BLEU-2:", bleu2)
print("Corpus BLEU-3:", bleu3)
print("Corpus BLEU-4:", bleu4)