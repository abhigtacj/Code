import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json
from tqdm import tqdm
from torchvision import transforms

# -------------------------
# CONFIG
# -------------------------

TRAIN_BATCH_SIZE = 1
VAL_BATCH_SIZE = 8
EPOCHS = 1
LR = 5e-5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# LOAD MODEL
# -------------------------

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(DEVICE)

# Freeze vision encoder for faster training
for param in model.vision_model.parameters():
    param.requires_grad = False

print("Vision encoder frozen")

# -------------------------
# IMAGE TRANSFORM
# -------------------------

image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -------------------------
# DATASET
# -------------------------

class CocoCaptionDataset(Dataset):

    def __init__(self, annotation_file, image_dir):

        with open(annotation_file) as f:
            coco = json.load(f)

        self.image_dir = image_dir

        self.image_map = {
            img["id"]: img["file_name"]
            for img in coco["images"]
        }

        self.annotations = coco["annotations"]

        captions = [ann["caption"] for ann in self.annotations]

        self.tokens = processor.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        ann = self.annotations[idx]

        image_id = ann["image_id"]

        filename = self.image_map[image_id]

        image_path = f"{self.image_dir}/{filename}"

        image = Image.open(image_path).convert("RGB")

        image = image_transform(image)

        input_ids = self.tokens["input_ids"][idx]
        attention_mask = self.tokens["attention_mask"][idx]

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

# -------------------------
# VALIDATION
# -------------------------

def validate(model, loader):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        progress = tqdm(loader, desc="Validation")

        for batch in progress:

            batch = {k:v.to(DEVICE) for k,v in batch.items()}

            with torch.cuda.amp.autocast():

                outputs = model(**batch)

            loss = outputs.loss

            total_loss += loss.item()

    return total_loss / len(loader)

# -------------------------
# MAIN TRAINING
# -------------------------

def main():

    train_dataset = CocoCaptionDataset(
        "coco_subset/annotations/captions_train2017_15k.json",
        "coco_subset/train2017_15k"
    )

    val_dataset = CocoCaptionDataset(
        "annotations/captions_val2017.json",
        "val2017"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch in progress:

            batch = {k:v.to(DEVICE) for k,v in batch.items()}

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                outputs = model(**batch)

                loss = outputs.loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            running_loss += loss.item()

            progress.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)

        val_loss = validate(model, val_loader)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        if val_loss < best_val:

            best_val = val_loss

            torch.save(model.state_dict(), "blip_attention_best.pth")

            print("Model saved")

    print("Training finished")

# -------------------------
# WINDOWS MULTIPROCESS FIX
# -------------------------

if __name__ == "__main__":
    main()