import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

    def generate_caption(self, features, vocab, max_length=20):
        """Greedy decoding"""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)  # (batch, 1, hidden)
            outputs = self.fc(hiddens.squeeze(1))        # (batch, vocab_size)
            _, predicted = outputs.max(1)                # greedy
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
            if predicted.item() == vocab["<end>"]:
                break
        return sampled_ids

# === Load Checkpoint ===
checkpoint = torch.load("flickr8k/cnn_lstm_model.pth", map_location=device)
vocab = checkpoint["vocab"]

embed_size = 256
hidden_size = 512
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, len(vocab)).to(device)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

# === Transformations ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Helper: Convert IDs to Words ===
inv_vocab = {idx: word for word, idx in vocab.items()}

def decode_caption(sampled_ids):
    words = []
    for idx in sampled_ids:
        word = inv_vocab.get(idx, "<unk>")
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            words.append(word)
    return " ".join(words)

# === Load Image Names from Excel ===
input_excel = "flickr8k/cleaned_file.xlsx"
df = pd.read_excel(input_excel)
image_names = df.iloc[:, 0].tolist()  # first column

results = []
total = len(image_names)

# === Process Images ===
for idx, img_name in enumerate(image_names, start=1):
    img_path = os.path.join("flickr8k/images", img_name)

    if not os.path.exists(img_path):
        caption = "Image not found"
    else:
        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = encoder(image_tensor)
            sampled_ids = decoder.generate_caption(features, vocab)
            caption = decode_caption(sampled_ids)

    results.append({"image_name": img_name, "caption": caption})

    # === Logging progress ===
    logging.info(f"[{idx}/{total}] Processed: {img_name}")
    print(f"[{idx}/{total}] Processed: {img_name}")

# === Save Results to Excel ===
output_excel = "flickr8k/captions_cnn_lstm.xlsx"
results_df = pd.DataFrame(results)
results_df.to_excel(output_excel, index=False)

print(f"\n✅ Captions saved to {output_excel}")