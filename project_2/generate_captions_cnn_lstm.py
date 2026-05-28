import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm

# ======================
# CONFIG
# ======================
FEATURE_DIR = "features_val"   # CNN features for val2017
ANNOTATION_FILE = "annotations/captions_val2017.json"
MODEL_PATH = "lstm_caption_model.pth"
OUTPUT_JSON = "cnn_lstm_val_captions.json"

EMBED_SIZE = 256
HIDDEN_SIZE = 512
MAX_LEN = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD COCO IMAGE IDS
# ======================
with open(ANNOTATION_FILE, "r") as f:
    coco = json.load(f)

image_id_to_file = {
    img["id"]: img["file_name"] for img in coco["images"]
}

# ======================
# MODEL DEFINITION
# ======================
class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.feature_fc = nn.Linear(2048, embed_size)  # IMPORTANT
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def sample(self, features, max_len=20):
        sampled_ids = []
        states = None

        # project CNN features
        inputs = self.feature_fc(features).unsqueeze(1)

        for _ in range(max_len):
            outputs, states = self.lstm(inputs, states)
            scores = self.fc(outputs.squeeze(1))
            predicted = scores.argmax(1)

            sampled_ids.append(predicted.item())

            if predicted.item() == END_IDX:
                break

            inputs = self.embedding(predicted).unsqueeze(1)

        return sampled_ids


# ======================
# LOAD CHECKPOINT
# ======================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

vocab = checkpoint["vocab"]
idx2word = {idx: word for word, idx in vocab.items()}

START_IDX = vocab["<START>"]
END_IDX = vocab["<END>"]

decoder = LSTMDecoder(
    EMBED_SIZE,
    HIDDEN_SIZE,
    len(vocab)
).to(DEVICE)

decoder.load_state_dict(checkpoint["decoder_state"])
decoder.eval()

print(f"✅ Model loaded | Vocab size: {len(vocab)}")

# ======================
# GENERATE CAPTIONS
# ======================
results = []

with torch.no_grad():
    for image_id, file_name in tqdm(
        image_id_to_file.items(),
        desc="Generating captions"
    ):
        feature_path = os.path.join(
            FEATURE_DIR,
            file_name.replace(".jpg", ".pt")
        )

        if not os.path.exists(feature_path):
            continue

        feature = torch.load(feature_path).unsqueeze(0).to(DEVICE)

        sampled_ids = decoder.sample(feature, MAX_LEN)

        words = []
        for idx in sampled_ids:
            word = idx2word.get(idx, "<UNK>")
            if word == "<END>":
                break
            words.append(word)

        caption = " ".join(words)

        results.append({
            "image_id": image_id,
            "caption": caption
        })

# ======================
# SAVE OUTPUT
# ======================
with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)

print(f"✅ Captions saved to {OUTPUT_JSON}")
