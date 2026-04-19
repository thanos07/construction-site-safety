#!/usr/bin/env python3
"""
Layer 1: Florence-2 captioning on ConstructionSite10k
Reads train.json + test.json, runs <DETAILED_CAPTION> on each image,
saves results to annotations/captions_florence2.json
Resume-safe: skips already-processed image_ids.
"""

import json, os, torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

BASE       = Path("/Data1/cse_24203016/construction_site/data/constructionsite10k")
MODEL_DIR  = Path("/Data1/cse_24203016/construction_site/models/florence2-base")
OUT_FILE   = BASE / "annotations/captions_florence2.json"
BATCH_SIZE = 8
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print(f"Loading Florence-2 from {MODEL_DIR} ...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, trust_remote_code=True, torch_dtype=torch.float16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True)
print("Model loaded ✅")

# Load all records
records = []
for split in ["train", "test"]:
    ann_path = BASE / f"annotations/{split}.json"
    with open(ann_path) as f:
        data = json.load(f)
    for r in data:
        r["_split"] = split
    records.extend(data)
print(f"Total records: {len(records)}")

# Load existing results (resume)
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        results = json.load(f)
    done_ids = {r["image_id"] for r in results}
    print(f"Resuming — {len(done_ids)} already done")
else:
    results  = []
    done_ids = set()

todo = [r for r in records if r["image_id"] not in done_ids]
print(f"To process: {len(todo)}")

SAVE_EVERY = 200

for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Captioning"):
    batch = todo[i : i + BATCH_SIZE]
    images, valid = [], []

    for r in batch:
        img_path = BASE / "images" / r["_split"] / r["file_name"]
        try:
            images.append(Image.open(img_path).convert("RGB"))
            valid.append(r)
        except Exception as e:
            print(f"  ⚠️  {r['file_name']}: {e}")

    if not images:
        continue

    inputs = processor(
        text=["<DETAILED_CAPTION>"] * len(images),
        images=images,
        return_tensors="pt",
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        generated = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"].to(torch.float16),
            max_new_tokens=128,
            do_sample=False,
        )

    captions = processor.batch_decode(generated, skip_special_tokens=True)

    for r, cap in zip(valid, captions):
        results.append({
            "image_id":        r["image_id"],
            "file_name":       r["file_name"],
            "split":           r["_split"],
            "gt_caption":      r["caption"],
            "florence2_caption": cap.strip(),
        })
        done_ids.add(r["image_id"])

    # Save checkpoint
    if len(results) % SAVE_EVERY < BATCH_SIZE:
        with open(OUT_FILE, "w") as f:
            json.dump(results, f)

# Final save
with open(OUT_FILE, "w") as f:
    json.dump(results, f)

print(f"\n✅ Done. {len(results)} captions saved to {OUT_FILE}")
