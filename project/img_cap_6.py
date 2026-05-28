import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset
from gtts import gTTS
import tempfile
import gradio as gr
import csv

# === Step 1: Load and Prepare Dataset ===
class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.samples = []

        with open(captions_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) == 2:
                    filename, caption = row
                    image_path = os.path.join(image_dir, filename.strip())
                    if os.path.exists(image_path):
                        self.samples.append((image_path, caption.strip()))
                    else:
                        print(f"[Missing Image] {image_path}")

        print(f"[Dataset Loaded] {len(self.samples)} valid image-caption pairs.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, caption = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=caption, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in inputs.items()}

# === Step 2: Load Base Model and Processor ===
base_model_name = "Salesforce/blip-image-captioning-base"
processor = AutoProcessor.from_pretrained(base_model_name)
model = BlipForConditionalGeneration.from_pretrained(base_model_name)

# === Step 3: Fine-Tune the Model ===
image_dir = "flickr8k/images"
captions_file = "flickr8k/captions_train.txt"  # CSV format with header
dataset = Flickr8kDataset(image_dir, captions_file, processor)

if len(dataset) == 0:
    raise ValueError("No valid samples found. Check image paths and captions file format.")

if __name__ == "__main__":
    from transformers import Trainer, TrainingArguments

    save_path = os.path.abspath("blip-flickr8k")

    training_args = TrainingArguments(
        output_dir=save_path,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        save_steps=500,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        fp16=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor
    )

    trainer.train()

    #  Save model and processor
    if not os.path.exists(os.path.join(save_path, "preprocessor_config.json")):
        raise FileNotFoundError("Missing preprocessor_config.json. Did you save the processor?")
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

    #  Reload after saving
    processor = AutoProcessor.from_pretrained(save_path)
    model = BlipForConditionalGeneration.from_pretrained(save_path)

    if not os.path.exists(os.path.join(save_path, "preprocessor_config.json")):
        raise FileNotFoundError("Missing preprocessor_config.json. Did you save the processor?")

# === Step 4: Reload Fine-Tuned Model for Inference ===
reload_path = os.path.abspath("blip-flickr8k")  # Use the actual save path

processor = AutoProcessor.from_pretrained(reload_path)
model = BlipForConditionalGeneration.from_pretrained(reload_path)

# === Step 5: Gradio App for Captioning + TTS ===
def caption_image(input_image: np.ndarray):
    image = Image.fromarray(input_image).convert('RGB')
    inputs = processor(images=image, text="a picture of", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50)
    caption = processor.decode(outputs[0], skip_special_tokens=True)

    tts = gTTS(text=caption)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_path = fp.name

    return caption, audio_path

iface = gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Text(label="Generated Caption"),
        gr.Audio(label="Listen to Caption", type="filepath")
    ],
    title="Image Captioning with TTS",
    description="Upload an image to generate a caption and hear it spoken aloud."
)

iface.launch()