import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

use_amp = torch.cuda.is_available()

# ---------------- Dataset ----------------
class CocoCaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor, max_length=20):
        self.image_dir = image_dir
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.anns.keys())
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ann_id = self.ids[idx]
        ann = self.coco.anns[ann_id]

        caption = ann["caption"]
        img_id = ann["image_id"]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

# ---------------- Training & Validation ----------------
def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, grad_accum_steps=8):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    progress_bar = tqdm(loader, desc="Training")

    for step, batch in enumerate(progress_bar):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        progress_bar.set_postfix(loss=loss.item() * grad_accum_steps)

    return total_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            total_loss += outputs.loss.item()
    return total_loss / len(loader)

# ---------------- Main ----------------
def main():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    for param in model.vision_model.parameters():
        param.requires_grad = False
    model.to(device)

    train_dataset = CocoCaptionDataset(
        image_dir="coco_subset/train2017_15k",
        annotation_file="coco_subset/annotations/captions_train2017_15k.json",
        processor=processor,
    )

    val_dataset = CocoCaptionDataset(
        image_dir="val2017",
        annotation_file="annotations/captions_val2017.json",
        processor=processor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5
    )

    scaler = torch.amp.GradScaler(enabled=use_amp)
    GRAD_ACCUM_STEPS = 8
    EPOCHS = 10
    PATIENCE = 1

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp, GRAD_ACCUM_STEPS)
        val_loss = validate(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            print("Validation improved. Saving best model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), "blip_best_model.pth")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement. Patience: {epochs_without_improvement}/{PATIENCE}")
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------- Entry Point ----------------
if __name__ == "__main__":
    main()