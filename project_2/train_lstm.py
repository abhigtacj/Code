import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# ======================
# NLTK SAFETY (RUN ONCE)
# ======================
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# ======================
# CONFIG
# ======================
FEATURE_DIR = "features"
CAPTION_FILE = "annotations/captions_train2017.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EMBED_SIZE = 256
HIDDEN_SIZE = 512
EPOCHS = 10
LR = 1e-3
MAX_LEN = 30

print("Using device:", DEVICE)

# ======================
# DATASET
# ======================
class CocoFeatureDataset(Dataset):
    def __init__(self, feature_dir, caption_file):
        with open(caption_file, "r") as f:
            coco = json.load(f)

        self.feature_dir = feature_dir
        self.annotations = coco["annotations"]
        self.images = {img["id"]: img["file_name"] for img in coco["images"]}

        counter = Counter()
        for ann in self.annotations:
            tokens = word_tokenize(ann["caption"].lower())
            counter.update(tokens)

        self.vocab = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        idx = 4
        for word, freq in counter.items():
            if freq >= 5:
                self.vocab[word] = idx
                idx += 1

    def encode_caption(self, caption):
        tokens = word_tokenize(caption.lower())
        encoded = [self.vocab.get(w, self.vocab["<UNK>"]) for w in tokens]
        encoded = [self.vocab["<START>"]] + encoded + [self.vocab["<END>"]]
        return encoded[:MAX_LEN]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        caption = ann["caption"]

        img_name = self.images[image_id]
        feature_path = os.path.join(
            self.feature_dir, img_name.replace(".jpg", ".pt")
        )

        feature = torch.load(feature_path)  # shape: (2048,)
        caption = torch.tensor(self.encode_caption(caption))

        return feature, caption

# ======================
# COLLATE FUNCTION
# ======================
def collate_fn(batch):
    features, captions = zip(*batch)
    features = torch.stack(features)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return features, padded, lengths

# ======================
# MODEL
# ======================
class LSTMDecoder(nn.Module):
    def __init__(self, feature_size, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.feature_fc = nn.Linear(feature_size, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        features = self.feature_fc(features)
        embeddings = self.embedding(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(embeddings)
        return self.fc(outputs)

# ======================
# MAIN TRAINING LOGIC
# ======================
def main():
    dataset = CocoFeatureDataset(FEATURE_DIR, CAPTION_FILE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,      # SAFE on Windows with __main__
        pin_memory=True
    )

    decoder = LSTMDecoder(
        feature_size=2048,
        embed_size=EMBED_SIZE,
        hidden_size=HIDDEN_SIZE,
        vocab_size=len(dataset.vocab)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=LR)

    print("Starting LSTM training...")

    for epoch in range(EPOCHS):
        decoder.train()
        total_loss = 0

        for features, captions, lengths in tqdm(
            loader, desc=f"Epoch {epoch+1}/{EPOCHS}"
        ):
            features = features.to(DEVICE)
            captions = captions.to(DEVICE)

            outputs = decoder(features, captions)
            targets = captions[:, 1:]
            outputs = outputs[:, :targets.size(1), :]

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Avg Loss: {total_loss / len(loader):.4f}")

    # ======================
    # SAVE MODEL
    # ======================
    torch.save(
        {
            "decoder_state": decoder.state_dict(),
            "vocab": dataset.vocab
        },
        "lstm_caption_model.pth"
    )

    print("Model saved as lstm_caption_model.pth")

# ======================
# WINDOWS ENTRY POINT
# ======================
if __name__ == "__main__":
    main()
