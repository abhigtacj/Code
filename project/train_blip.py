import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    TrainingArguments,
    Trainer
)

# === Custom Trainer to ensure loss is computed ===
class BlipTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Ensure 'labels' exist safely
        if "labels" not in inputs:
            inputs["labels"] = inputs["input_ids"].clone()

        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# === Custom Collator for BLIP ===
class BlipCollator:
    def __init__(self, processor):
        self.tokenizer = processor.tokenizer

    def __call__(self, batch):
        image_inputs = [b["pixel_values"] for b in batch]
        text_inputs = [{"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]} for b in batch]

        # Pad the text inputs
        padded = self.tokenizer.pad(
            text_inputs,
            padding=True,
            return_tensors="pt"
        )

        # Add images
        padded["pixel_values"] = torch.stack(image_inputs)

        # Add labels for loss computation
        # (clone input_ids so model learns to predict captions)
        padded["labels"] = padded["input_ids"].clone()

        return padded


# === Flickr8k Dataset Loader ===
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

        # Processor handles both text and image
        inputs = self.processor(images=image, text=caption, return_tensors="pt")

        # Remove extra batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Ensure labels are present
        inputs["labels"] = inputs["input_ids"].clone()

        return inputs


# === Training Script ===
if __name__ == "__main__":
    image_dir = "flickr8k/images"
    captions_file = "flickr8k/captions_train.txt"
    save_path = os.path.abspath("blip-flickr8k")

    # Load processor and model
    processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load dataset
    dataset = Flickr8kDataset(image_dir, captions_file, processor)

    # Training configuration
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

    # Collator
    data_collator = BlipCollator(processor)

    # Trainer
    trainer = BlipTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        processing_class=processor
    )

    # Train
    trainer.train()

    # Save model and processor
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
