import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from collections import Counter

# ===============================
# DEVICE
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# VOCAB CLASS
# ===============================
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold

        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return text.lower().split()

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized
        ]


# ===============================
# DATASET
# ===============================
class CustomDataset(Dataset):
    def __init__(self, image_folder, captions_file, transform=None, freq_threshold=5):

        self.image_folder = image_folder
        self.transform = transform

        with open(captions_file, "r") as f:
            data = json.load(f)

        self.images = []
        self.captions = []

        for ann in data["annotations"]:
            self.images.append(ann["image_id"])
            self.captions.append(ann["caption"])

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions)

        self.image_id_to_filename = {
            img["id"]: img["file_name"]
            for img in data["images"]
        }

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        image_id = self.images[index]

        img_path = os.path.join(
            self.image_folder,
            self.image_id_to_filename[image_id]
        )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<start>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized_caption)


# ===============================
# COLLATE FUNCTION (PAD)
# ===============================
def collate_fn(batch):
    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    images = torch.stack(images)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)

    padded = torch.zeros(len(captions), max_len).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        padded[i, :end] = cap

    return images, padded


# ===============================
# ENCODER
# ===============================
class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

        for p in self.resnet.parameters():
            p.requires_grad = False

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.permute(0, 2, 3, 1)
        return features


# ===============================
# ATTENTION
# ===============================
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):

        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)

        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding


# ===============================
# DECODER
# ===============================
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.attention = Attention(2048, 512, 256)
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTMCell(256 + 2048, 512)
        self.fc = nn.Linear(512, vocab_size)
        self.dropout = nn.Dropout(0.5)

        self.init_h = nn.Linear(2048, 512)
        self.init_c = nn.Linear(2048, 512)

        self.vocab_size = vocab_size

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions):

        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.view(batch_size, -1, 2048)

        embeddings = self.embedding(captions)

        h, c = self.init_hidden_state(encoder_out)

        seq_len = captions.size(1)
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)

        for t in range(seq_len):

            attention_weighted_encoding = self.attention(encoder_out, h)

            lstm_input = torch.cat(
                [embeddings[:, t, :], attention_weighted_encoding],
                dim=1
            )

            h, c = self.lstm(lstm_input, (h, c))
            preds = self.fc(self.dropout(h))

            outputs[:, t, :] = preds

        return outputs


# ===============================
# TRAINING
# ===============================
def train():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = CustomDataset(
        image_folder="train2017",
        captions_file="annotations/captions_train2017.json",
        transform=transform
    )

    vocab_size = len(dataset.vocab)
    pad_idx = dataset.vocab.stoi["<pad>"]

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )


    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(decoder.parameters(), lr=1e-4)

    for epoch in range(10):

        loop = tqdm(loader, desc=f"Epoch [{epoch+1}/10]")
        total_loss = 0

        for images, captions in loop:

            images = images.to(device)
            captions = captions.to(device)

            optimizer.zero_grad()

            encoder_out = encoder(images)
            outputs = decoder(encoder_out, captions[:, :-1])

            loss = criterion(
                outputs.reshape(-1, vocab_size),
                captions[:, 1:].reshape(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(loader)}")

    torch.save({
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "vocab": dataset.vocab
    }, "attention_full_model.pth")

    print("Model saved as attention_full_model.pth")


if __name__ == "__main__":
    train()
