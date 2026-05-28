import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm

# ======================
# CONFIG
# ======================
IMAGE_DIR = "train2017"
CAPTION_FILE = "annotations/captions_train2017.json"
MODEL_SAVE_PATH = "cnn_lstm_coco.pth"

BATCH_SIZE = 32        # safe for 1650 Ti
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_EPOCHS = 10
LR = 1e-3
MAX_LEN = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# ======================
# DATASET
# ======================
class CocoDataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        with open(caption_file, "r") as f:
            coco = json.load(f)

        self.image_dir = image_dir
        self.transform = transform
        self.annotations = coco["annotations"]
        self.images = {img["id"]: img["file_name"] for img in coco["images"]}

        # Build vocabulary
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

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def __len__(self):
        return len(self.annotations)

    def encode_caption(self, caption):
        tokens = word_tokenize(caption.lower())
        encoded = [self.vocab.get(word, self.vocab["<UNK>"]) for word in tokens]
        encoded = [self.vocab["<START>"]] + encoded + [self.vocab["<END>"]]
        return encoded[:MAX_LEN]

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        caption = ann["caption"]

        image_path = os.path.join(self.image_dir, self.images[image_id])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption_encoded = self.encode_caption(caption)
        caption_tensor = torch.tensor(caption_encoded)

        return image, caption_tensor


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)

    lengths = [len(c) for c in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len).long()
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return images, padded, lengths

# ======================
# MODELS
# ======================
class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.squeeze()
        features = self.fc(features)
        return features


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(embeddings)
        outputs = self.fc(outputs)
        return outputs

# ======================
# TRAINING
# ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = CocoDataset(IMAGE_DIR, CAPTION_FILE, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                    shuffle=True, collate_fn=collate_fn)

encoder = CNNEncoder(EMBED_SIZE).to(DEVICE)
decoder = LSTMDecoder(EMBED_SIZE, HIDDEN_SIZE, len(dataset.vocab)).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(decoder.parameters(), lr=LR)

print("Starting training...")

for epoch in range(NUM_EPOCHS):
    encoder.eval()
    decoder.train()

    total_loss = 0
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

    for images, captions, lengths in loop:
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        features = encoder(images)
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
        loop.set_postfix(loss=loss.item())

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Avg Loss: {total_loss/len(loader):.4f}")

# ======================
# SAVE MODEL
# ======================
torch.save({
    "encoder": encoder.state_dict(),
    "decoder": decoder.state_dict(),
    "vocab": dataset.vocab
}, MODEL_SAVE_PATH)

print("Model saved to:", MODEL_SAVE_PATH)
