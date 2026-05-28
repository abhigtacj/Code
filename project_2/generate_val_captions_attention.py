import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json
from tqdm import tqdm
from collections import Counter
from pycocotools.coco import COCO

# =========================================================
# Device
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# =========================================================
# Encoder
# =========================================================

class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
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


# =========================================================
# Attention
# =========================================================

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


# =========================================================
# Decoder
# =========================================================

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

    def generate_caption(self, encoder_out, vocab, max_len=30, device=device):
        """
        Greedy decode for a single image (batch_size == 1).
        Returns a list of token ids (excluding the <start> token).
        """
        self.eval()
        with torch.no_grad():
            # Expect encoder_out shape: (1, H, W, 2048) as produced by EncoderCNN
            batch_size = encoder_out.size(0)
            assert batch_size == 1, "generate_caption currently supports batch_size=1"

            encoder_out = encoder_out.view(batch_size, -1, 2048)  # (1, num_pixels, 2048)

            # initialize hidden state
            h, c = self.init_hidden_state(encoder_out)  # (1, 512)

            # start token
            start_idx = vocab.stoi.get("<start>", 1)
            end_idx = vocab.stoi.get("<end>", 2)

            inputs = torch.tensor([start_idx], dtype=torch.long, device=device).unsqueeze(0)  # (1,1)
            inputs = inputs.squeeze(1)  # (1,)

            generated_ids = []

            for _ in range(max_len):
                # embedding for current input token
                emb = self.embedding(inputs)  # (1, embed_dim)

                # attention
                attention_weighted_encoding = self.attention(encoder_out, h)  # (1, 2048)

                lstm_input = torch.cat([emb, attention_weighted_encoding], dim=1)  # (1, 256+2048)

                h, c = self.lstm(lstm_input, (h, c))  # (1, 512)

                preds = self.fc(self.dropout(h))  # (1, vocab_size)
                predicted = preds.argmax(dim=1)  # (1,)

                token_id = predicted.item()
                if token_id == end_idx:
                    break

                generated_ids.append(token_id)

                # next input
                inputs = predicted

            return generated_ids

# =========================================================
# COCO Val Dataset
# =========================================================

class CocoValDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        path = img_info["file_name"]

        image = Image.open(os.path.join(self.image_dir, path)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_id


# =========================================================
# Paths (EDIT THESE)
# =========================================================

MODEL_PATH = "attention_full_model.pth"
VAL_IMAGE_DIR = "val2017"
VAL_ANNOTATION = "annotations/captions_val2017.json"
OUTPUT_JSON = "cnn_lstm_val_attention_captions.json"

# =========================================================
# Load checkpoint
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False   # 🔥 IMPORTANT
)

vocab = checkpoint["vocab"]
vocab_size = len(vocab)

encoder = EncoderCNN().to(device)
decoder = DecoderRNN(vocab_size).to(device)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

print("Model loaded successfully.")

# Reverse vocab
rev_vocab = vocab.itos

# =========================================================
# Transform
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

dataset = CocoValDataset(VAL_IMAGE_DIR, VAL_ANNOTATION, transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# =========================================================
# Generate Captions
# =========================================================

results = []

print("Generating captions...")

with torch.no_grad():
    for images, img_id in tqdm(loader):

        images = images.to(device)

        encoder_out = encoder(images)
        caption_ids = decoder.generate_caption(encoder_out, vocab)

        words = [rev_vocab[idx] for idx in caption_ids]
        sentence = " ".join(words)

        results.append({
            "image_id": int(img_id.item()),
            "caption": sentence
        })

# =========================================================
# Save JSON
# =========================================================

with open(OUTPUT_JSON, "w") as f:
    json.dump(results, f)

print("Saved to", OUTPUT_JSON)
