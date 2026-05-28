import nltk
nltk.download("punkt")
import json
import torch
from collections import Counter
from nltk.tokenize import word_tokenize

# ======================
# CONFIG
# ======================
ANNOTATION_FILE = "annotations/captions_train2017.json"
VOCAB_PATH = "vocab.pth"
MIN_FREQ = 5

# ======================
# LOAD COCO CAPTIONS
# ======================
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

annotations = coco["annotations"]

# ======================
# BUILD WORD COUNTS
# ======================
counter = Counter()

for ann in annotations:
    caption = ann["caption"].lower()
    tokens = word_tokenize(caption)
    counter.update(tokens)

# ======================
# BUILD VOCAB
# ======================
vocab = {
    "<PAD>": 0,
    "<START>": 1,
    "<END>": 2,
    "<UNK>": 3
}

idx = 4
for word, freq in counter.items():
    if freq >= MIN_FREQ:
        vocab[word] = idx
        idx += 1

# ======================
# SAVE
# ======================
torch.save(vocab, VOCAB_PATH)

print(f"✅ Vocabulary built and saved")
print(f"Total vocab size: {len(vocab)}")
print(f"Saved to: {VOCAB_PATH}")
