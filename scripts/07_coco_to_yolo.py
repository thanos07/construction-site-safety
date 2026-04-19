#!/usr/bin/env python3
import json, os, shutil, random
from pathlib import Path
import yaml

BASE      = "/Data1/cse_24203016/construction_site"
COCO_JSON = f"{BASE}/data/constructionsite10k/annotations/pseudo_labels_coco.json"  # FIXED path
IMAGE_DIR = f"{BASE}/data/constructionsite10k/images"
DATASET   = f"{BASE}/datasets/construction"
LABEL_TMP = f"{BASE}/datasets/yolo_labels_tmp"

with open(COCO_JSON) as f:
    coco = json.load(f)

classes = [c["name"] for c in coco["categories"]]
img_map = {img["id"]: img for img in coco["images"]}
anns_by = {img["id"]: [] for img in coco["images"]}
for ann in coco["annotations"]:
    anns_by[ann["image_id"]].append(ann)

os.makedirs(f"{LABEL_TMP}", exist_ok=True)
labeled_imgs = []

for img_id, anns in anns_by.items():
    if not anns:
        continue
    img_info = img_map[img_id]
    W, H     = img_info["width"], img_info["height"]
    fname    = img_info["file_name"]
    split    = img_info.get("split", "train")
    stem     = Path(fname).stem

    lines = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        cx = (x + w/2) / W;  cy = (y + h/2) / H
        nw = w / W;           nh = h / H
        cls_id = ann["category_id"] - 1   # FIXED: COCO is 1-indexed, YOLO needs 0-indexed
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    lbl_path = f"{LABEL_TMP}/{stem}.txt"
    with open(lbl_path, "w") as f:
        f.write("\n".join(lines))

    img_path = Path(IMAGE_DIR) / split / fname
    if img_path.exists():
        labeled_imgs.append((img_path, lbl_path))
    else:
        # fallback search
        hits = list(Path(IMAGE_DIR).rglob(fname))
        if hits:
            labeled_imgs.append((hits[0], lbl_path))

print(f"Images with labels: {len(labeled_imgs)}")

# 80/10/10 split
random.seed(42)
random.shuffle(labeled_imgs)
n = len(labeled_imgs)
splits = {
    "train": labeled_imgs[:int(0.8*n)],
    "val":   labeled_imgs[int(0.8*n):int(0.9*n)],
    "test":  labeled_imgs[int(0.9*n):]
}

for split, pairs in splits.items():
    os.makedirs(f"{DATASET}/{split}/images", exist_ok=True)
    os.makedirs(f"{DATASET}/{split}/labels", exist_ok=True)
    for img_path, lbl_path in pairs:
        shutil.copy(img_path, f"{DATASET}/{split}/images/{Path(img_path).name}")
        shutil.copy(lbl_path, f"{DATASET}/{split}/labels/{Path(lbl_path).name}")

yaml_cfg = {
    "path":  DATASET,
    "train": "train/images",
    "val":   "val/images",
    "test":  "test/images",
    "nc":    len(classes),
    "names": classes
}
yaml_path = f"{BASE}/datasets/construction.yaml"
with open(yaml_path, "w") as f:
    yaml.dump(yaml_cfg, f, default_flow_style=False)

print(f"Split: {len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test")
print(f"Classes ({len(classes)}): {classes}")
print(f"YAML: {yaml_path}")
