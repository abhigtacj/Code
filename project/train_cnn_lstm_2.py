import os
import re
import csv
import math
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pandas as pd
from functools import partial

# =========================
# Paths
# =========================
IMAGE_DIR = "flickr8k/images"
CAPTIONS_PATH = "flickr8k/captions_train.txt"  # supports .csv or .txt
SAVE_DIR = "flickr8k/artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "cnn_lstm_model.pth")

# =========================
# Logging setup
# =========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/training_cnn_lstm.log", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# =========================
# Device setup (CPU/GPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# =========================
# Config
# =========================
EPOCHS = 15
BATCH_SIZE = 32
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
LEARNING_RATE = 1e-3
VALID_SPLIT = 0.1
MIN_FREQ = 1
MAX_CAPTION_LEN = 25

# =========================
# Utilities
# =========================
def load_captions(path: str):
    rows = []
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    rows.append((row[0].strip(), row[1].strip()))
    else:
        # assume delimited text with header; pandas auto-detects separator
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] < 2:
            raise ValueError("Captions file must have at least two columns: image_name, caption.")
        rows = [(str(df.iloc[i, 0]).strip(), str(df.iloc[i, 1]).strip()) for i in range(len(df))]
    return rows

def tokenize(text: str):
    text = text.lower().strip()
    return [t for t in re.split(r"\W+", text) if t]

def build_vocab(pairs, min_freq=MIN_FREQ):
    freq = {}
    for _, caption in pairs:
        for w in tokenize(caption)[:MAX_CAPTION_LEN]:
            freq[w] = freq.get(w, 0) + 1
    vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    idx = 4
    for w, c in freq.items():
        if c >= min_freq and w not in vocab:
            vocab[w] = idx
            idx += 1
    logging.info(f"Vocabulary size: {len(vocab)}")
    return vocab

def encode_caption(caption, vocab):
    ids = [vocab["<start>"]]
    for w in tokenize(caption)[:MAX_CAPTION_LEN]:
        ids.append(vocab.get(w, vocab["<unk>"]))
    ids.append(vocab["<end>"])
    return ids

def pad_sequence(sequences, pad_value):
    lengths = [s.size(0) for s in sequences]
    max_len = max(lengths)
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(sequences):
        padded[i, : s.size(0)] = s
    return padded

# =========================
# Dataset
# =========================
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, pairs, vocab, transform=None):
        self.image_dir = image_dir
        self.samples = []
        self.vocab = vocab
        self.transform = transform
        for filename, caption in pairs:
            img_path = os.path.join(image_dir, filename)
            if os.path.exists(img_path):
                self.samples.append((img_path, caption))
            else:
                logging.warning(f"Missing image: {img_path}")
        logging.info(f"Loaded {len(self.samples)} image-caption pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, caption = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        cap_ids = encode_caption(caption, self.vocab)
        cap_tensor = torch.tensor(cap_ids, dtype=torch.long)
        return image, cap_tensor

def collate_batch(batch, pad_value):
    images = torch.stack([b[0] for b in batch])
    caps = [b[1] for b in batch]
    caps_padded = pad_sequence(caps, pad_value=pad_value)
    return images, caps_padded

# =========================
# Model
# =========================
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        features = self.bn(features)
        features = self.dropout(features)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        hiddens = self.dropout(hiddens)
        outputs = self.fc(hiddens)
        return outputs

# =========================
# Training loops
# =========================
def train_one_epoch(encoder, decoder, loader, criterion, optimizer, pad_idx):
    encoder.train()
    decoder.train()
    running = 0.0
    for images, captions in tqdm(loader, desc="Train", leave=False):
        images, captions = images.to(device), captions.to(device)

        inputs_captions = captions[:, :-1]
        targets = captions[:, 1:]

        features = encoder(images)
        outputs = decoder(features, inputs_captions)
        outputs = outputs[:, :-1, :]

        loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()
    return running / len(loader)

@torch.no_grad()
def evaluate(encoder, decoder, loader, criterion, pad_idx):
    encoder.eval()
    decoder.eval()
    running = 0.0
    for images, captions in tqdm(loader, desc="Val", leave=False):
        images, captions = images.to(device), captions.to(device)

        inputs_captions = captions[:, :-1]
        targets = captions[:, 1:]

        features = encoder(images)
        outputs = decoder(features, inputs_captions)
        outputs = outputs[:, :-1, :]

        loss = criterion(outputs.reshape(-1, outputs.size(2)), targets.reshape(-1))
        running += loss.item()
    return running / len(loader)

# =========================
# Main
# =========================
def main():
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load data
    pairs = load_captions(CAPTIONS_PATH)
    if len(pairs) == 0:
        raise RuntimeError("No caption pairs found. Check your captions file path and format.")

    # Build vocab and dataset
    vocab = build_vocab(pairs, min_freq=MIN_FREQ)
    dataset = Flickr8kDataset(IMAGE_DIR, pairs, vocab, transform)

    # Split into train/val
    val_size = max(1, int(math.ceil(len(dataset) * VALID_SPLIT)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    # Dataloaders (use picklable collate with functools.partial)
    pad_idx = vocab["<pad>"]
    collate_fn_train = partial(collate_batch, pad_value=pad_idx)
    collate_fn_val = partial(collate_batch, pad_value=pad_idx)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn_train, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn_val, num_workers=2, pin_memory=True
    )

    # Models
    encoder = EncoderCNN(EMBED_SIZE, dropout=0.1).to(device)
    decoder = DecoderRNN(EMBED_SIZE, HIDDEN_SIZE, len(vocab), NUM_LAYERS, dropout=0.3).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(
        [
            {"params": encoder.parameters(), "lr": LEARNING_RATE * 0.5},  # smaller LR for encoder
            {"params": decoder.parameters(), "lr": LEARNING_RATE}
        ]
    )

    best_val = float("inf")
    logging.info(f"Dataset size: {len(dataset)} | Train: {train_size} | Val: {val_size}")
    logging.info(f"Saving checkpoints to: {CHECKPOINT_PATH}")

    # Training epochs
    for epoch in range(1, EPOCHS + 1):
        logging.info(f"Epoch {epoch}/{EPOCHS} started")

        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion, optimizer, pad_idx)
        val_loss = evaluate(encoder, decoder, val_loader, criterion, pad_idx)

        logging.info(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict(),
                    "vocab": vocab,
                    "config": {
                        "embed_size": EMBED_SIZE,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "normalize": True
                    }
                },
                CHECKPOINT_PATH
            )
            logging.info(f"Checkpoint saved (best val loss: {best_val:.4f})")

    # Final save (last epoch)
    final_path = os.path.join(SAVE_DIR, "cnn_lstm_model_last.pth")
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocab": vocab,
            "config": {
                "embed_size": EMBED_SIZE,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "normalize": True
            }
        },
        final_path
    )
    logging.info(f"Final model saved at: {final_path}")
    print(f"Training complete. Best model: {CHECKPOINT_PATH} | Last epoch: {final_path}")

if __name__ == "__main__":
    main()