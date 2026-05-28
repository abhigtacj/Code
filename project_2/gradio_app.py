import gradio as gr
import torch
from PIL import Image
import pyttsx3
import torchvision.transforms as transforms
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn as nn
import torchvision.models as models

# ============================
# DEVICE
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
# LOAD BLIP MODELS
# ============================

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_model.load_state_dict(torch.load("blip_best_model.pth", map_location=device))
blip_model.to(device).eval()

blip_att_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
blip_att_model.load_state_dict(torch.load("blip_attention_best.pth", map_location=device))
blip_att_model.to(device).eval()

# ============================
# CNN + LSTM (NO ATTENTION)
# ============================

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super().__init__()
        self.feature_fc = nn.Linear(2048, embed_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def sample(self, features, max_len=20):
        sampled_ids = []
        states = None
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

# Load CNN-LSTM
checkpoint_lstm = torch.load("lstm_caption_model.pth", map_location=device)
vocab_lstm = checkpoint_lstm["vocab"]
idx2word_lstm = {idx: word for word, idx in vocab_lstm.items()}
START_IDX = vocab_lstm["<START>"]
END_IDX = vocab_lstm["<END>"]

lstm_model = LSTMDecoder(256, 512, len(vocab_lstm)).to(device)
lstm_model.load_state_dict(checkpoint_lstm["decoder_state"])
lstm_model.eval()

# Dummy CNN feature extractor (ResNet50)
cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
cnn = nn.Sequential(*list(cnn.children())[:-1]).to(device)
cnn.eval()

# ============================
# VOCAB CLASS (REQUIRED FOR LOADING CHECKPOINT)
# ============================

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

# ============================
# CNN + LSTM WITH ATTENTION
# ============================

class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        return features.permute(0, 2, 3, 1)

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
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.attention = Attention(2048, 512, 256)
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTMCell(256 + 2048, 512)
        self.fc = nn.Linear(512, vocab_size)
        self.init_h = nn.Linear(2048, 512)
        self.init_c = nn.Linear(2048, 512)

    def generate_caption(self, encoder_out, vocab, max_len=30):
        encoder_out = encoder_out.view(1, -1, 2048)
        h = self.init_h(encoder_out.mean(dim=1))
        c = self.init_c(encoder_out.mean(dim=1))

        start_idx = vocab.stoi["<start>"]
        end_idx = vocab.stoi["<end>"]

        inputs = torch.tensor([start_idx]).to(device)
        generated = []

        for _ in range(max_len):
            emb = self.embedding(inputs)
            context = self.attention(encoder_out, h)
            lstm_input = torch.cat([emb, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            preds = self.fc(h)
            predicted = preds.argmax(1)

            token = predicted.item()
            if token == end_idx:
                break

            generated.append(token)
            inputs = predicted

        return generated

# Load attention model
checkpoint_att = torch.load("attention_full_model.pth", map_location=device, weights_only=False)
vocab_att = checkpoint_att["vocab"]
rev_vocab_att = vocab_att.itos

encoder = EncoderCNN().to(device)
decoder = DecoderRNN(len(vocab_att)).to(device)
encoder.load_state_dict(checkpoint_att["encoder"])
decoder.load_state_dict(checkpoint_att["decoder"])
encoder.eval()
decoder.eval()

# ============================
# TRANSFORM
# ============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# ============================
# GENERATE CAPTION
# ============================

def generate_caption(image, model_name):
    image = image.convert("RGB")

    if model_name == "BLIP":
        inputs = processor(image, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs, max_length=20)
        caption = processor.decode(output[0], skip_special_tokens=True)

    elif model_name == "BLIP_Attention":
        inputs = processor(image, return_tensors="pt").to(device)
        output = blip_att_model.generate(**inputs, max_length=30, num_beams=3)
        caption = processor.decode(output[0], skip_special_tokens=True)

    elif model_name == "CNN_LSTM":
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            # EXACT same feature extraction as your original script
            features = cnn(img_tensor)            # (1,2048,1,1)
            features = features.squeeze()         # (2048,)  ✅ EXACT MATCH
            features = features.unsqueeze(0)      # (1,2048)

            sampled_ids = lstm_model.sample(features)

        words = []
        for idx in sampled_ids:
            word = idx2word_lstm.get(idx, "<UNK>")
            if word == "<END>":
                break
            words.append(word)

        caption = " ".join(words)

    elif model_name == "CNN_LSTM_Attention":
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            enc_out = encoder(img_tensor)
            ids = decoder.generate_caption(enc_out, vocab_att)

        words = [rev_vocab_att[idx] for idx in ids]
        caption = " ".join(words)

    else:
        caption = "Invalid model selection"

    # TTS
    engine = pyttsx3.init()
    engine.say(caption)
    engine.runAndWait()

    return caption

# ============================
# GRADIO UI
# ============================

with gr.Blocks() as demo:
    gr.Markdown("# Image Captioning App with TTS")

    with gr.Row():
        image_input = gr.Image(type="pil")
        model_choice = gr.Dropdown(
            choices=["CNN_LSTM", "CNN_LSTM_Attention", "BLIP", "BLIP_Attention"],
            value="BLIP",
            label="Select Model"
        )

    output = gr.Textbox(label="Generated Caption")

    btn = gr.Button("Generate Caption")

    btn.click(generate_caption, inputs=[image_input, model_choice], outputs=output)

if __name__ == "__main__":
    demo.launch()
