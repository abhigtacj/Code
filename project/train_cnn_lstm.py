import os
import csv
import logging
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

# === Logging Setup ===
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Device Setup (CPU + GPU) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Dataset Loader ===
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, vocab, transform=None):
        self.image_dir = image_dir
        self.samples = []
        self.vocab = vocab
        self.transform = transform

        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if len(row) == 2:
                    filename, caption = row
                    image_path = os.path.join(image_dir, filename.strip())
                    if os.path.exists(image_path):
                        self.samples.append((image_path, caption.strip()))

        logging.info(f"Loaded {len(self.samples)} image-caption pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Convert caption to indices
        tokens = caption.lower().split()
        caption_ids = [self.vocab.get(word, self.vocab["<unk>"]) for word in tokens]
        caption_ids = [self.vocab["<start>"]] + caption_ids + [self.vocab["<end>"]]
        return image, torch.tensor(caption_ids)

# === Vocabulary Builder ===
def build_vocab(captions_file, min_freq=1):
    word_freq = {}
    with open(captions_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) == 2:
                _, caption = row
                for word in caption.lower().split():
                    word_freq[word] = word_freq.get(word, 0) + 1

    vocab = {"<pad>": 0, "<start>": 1, "<end>": 2, "<unk>": 3}
    idx = 4
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    logging.info(f"Vocabulary size: {len(vocab)}")
    return vocab

# === CNN Encoder ===
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove last FC
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# === LSTM Decoder ===
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.fc(hiddens)
        return outputs

# === Training Function ===
def train_model(encoder, decoder, dataloader, criterion, optimizer, num_epochs, save_path):
    encoder.train()
    decoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for images, captions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, captions = images.to(device), captions.to(device)

            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])  # input: all but last token

            # Drop the extra step from outputs to match target length
            outputs = outputs[:, :-1, :]  

            # Compute loss against shifted captions
            loss = criterion(
                outputs.reshape(-1, outputs.size(2)),
                captions[:, 1:].reshape(-1)
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save model checkpoint
    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "vocab": vocab
    }, save_path)
    logging.info(f"Model saved at {save_path}")
    print(f"Model saved at {save_path}")

# === Paths ===
image_dir = "flickr8k/images"
captions_file = "flickr8k/captions_train.txt"
save_path = "flickr8k/cnn_lstm_model.pth"

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Build Vocab and Dataset ===
vocab = build_vocab(captions_file)
dataset = Flickr8kDataset(image_dir, captions_file, vocab, transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: (
    torch.stack([i[0] for i in x]),
    nn.utils.rnn.pad_sequence([i[1] for i in x], batch_first=True, padding_value=vocab["<pad>"])
))

# === Model Setup ===
embed_size = 256
hidden_size = 512
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])
optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.fc.parameters()), lr=0.001)

# === Train ===
train_model(encoder, decoder, dataloader, criterion, optimizer, num_epochs=3, save_path=save_path)