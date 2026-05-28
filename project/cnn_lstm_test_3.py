import os
import logging
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

 # Paths
 
MODEL_PATH = "flickr8k/artifacts/cnn_lstm_model_last.pth"
IMAGE_DIR = "flickr8k/images"
INPUT_EXCEL = "flickr8k/cleaned_file.xlsx"
OUTPUT_EXCEL = "captions_cnn_lstm_3.xlsx"
 
# Logging setup
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

 # Device setup
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
 
# Model definitions
 
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

    def generate_caption(self, features, vocab, max_length=20):
        """Greedy decoding"""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None
        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.fc(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)
            if predicted.item() == vocab["<end>"]:
                break
        return sampled_ids

 # Load checkpoint
 
checkpoint = torch.load(MODEL_PATH, map_location=device)
vocab = checkpoint["vocab"]
config = checkpoint["config"]

encoder = EncoderCNN(config["embed_size"]).to(device)
decoder = DecoderRNN(config["embed_size"], config["hidden_size"],
                     len(vocab), config["num_layers"]).to(device)

encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

encoder.eval()
decoder.eval()

 # Transformations
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

 # Helper: decode caption
 
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

 # Load image names from Excel
 
df = pd.read_excel(INPUT_EXCEL)
image_names = df.iloc[:, 0].tolist()

results = []
total = len(image_names)

 # Process images
 
for idx, img_name in enumerate(image_names, start=1):
    img_path = os.path.join(IMAGE_DIR, img_name)

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

    # Logging progress
    logging.info(f"[{idx}/{total}] Processed: {img_name}")

 # Save results to Excel
 
results_df = pd.DataFrame(results)
results_df.to_excel(OUTPUT_EXCEL, index=False)

print(f"\n Captions saved to {OUTPUT_EXCEL}")