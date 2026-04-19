#!/usr/bin/env python3
import torch, json, os
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm

BASE      = "/Data1/cse_24203016/construction_site"
MODEL_DIR = f"{BASE}/models/florence2-base"
CAP_JSON  = f"{BASE}/data/constructionsite10k/annotations/captions_florence2.json"
IMG_BASE  = Path(f"{BASE}/data/constructionsite10k/images")
SAVE_DIR  = f"{BASE}/models/florence2-lora-finetuned"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Model path: {MODEL_DIR}")

processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)

class CaptionDataset(Dataset):
    def __init__(self, cap_json, img_base, split="train", max_n=None):
        with open(cap_json) as f:
            raw = json.load(f)
        self.data = [r for r in raw if r.get("split", "train") == split]
        if max_n:
            self.data = self.data[:max_n]
        self.img_base = Path(img_base)
        print(f"  {split} samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        r      = self.data[idx]
        caption = r["gt_caption"]
        split   = r.get("split", "train")
        fname   = r["file_name"]

        img_path = self.img_base / split / fname
        if not img_path.exists():
            hits = list(self.img_base.rglob(fname))
            img_path = hits[0] if hits else img_path

        img = Image.open(img_path).convert("RGB")

        # RULE: pass ONLY text + images to processor — NO padding/truncation/max_length
        # Those args break Florence-2's internal BART tokenizer
        enc = processor(
            text="<DETAILED_CAPTION>",
            images=img,
            return_tensors="pt",
        )

        # Labels: tokenize caption separately via tokenizer (NOT processor)
        label_ids = processor.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
            truncation=True,
        )["input_ids"].squeeze(0)

        # Mask padding so it doesn't contribute to loss
        label_ids[label_ids == processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids":    enc["input_ids"].squeeze(0),
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels":       label_ids,
        }

print("Loading Florence-2 model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float16, trust_remote_code=True
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model = model.to(device)

print("Loading dataset...")
train_ds = CaptionDataset(CAP_JSON, IMG_BASE, split="train")
val_ds   = CaptionDataset(CAP_JSON, IMG_BASE, split="test", max_n=500)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False, num_workers=0)

# num_workers=0 avoids all DataLoader worker subprocess issues

opt    = torch.amp.GradScaler("cuda")
scaler = torch.amp.GradScaler("cuda")
optim  = torch.optim.AdamW(model.parameters(), lr=2e-4)

EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [train]"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda"):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(optim); scaler.update(); optim.zero_grad()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [val]"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                loss = model(**batch).loss
            val_loss += loss.item()

    print(f"Epoch {epoch+1} — train: {train_loss/len(train_loader):.4f} | val: {val_loss/len(val_loader):.4f}")

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)
print(f"\n✅ LoRA model saved to {SAVE_DIR}")
