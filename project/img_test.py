import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import Dataset
#from gtts import gTTS
#import tempfile
#import gradio as gr

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []

        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '\t' in line:
                    filename, caption = line.strip().split('\t')
                    image_path = os.path.join(image_dir, filename)
                    if os.path.exists(image_path):
                        self.samples.append((image_path, caption))
                    else:
                        print(f"[Missing Image] {image_path}")

        print(f"[Dataset Loaded] {len(self.samples)} valid samples found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=caption, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
dataset = Flickr8kDataset("flickr8k/images", "flickr8k/captions.txt", processor)
print(f"Total samples: {len(dataset)}")