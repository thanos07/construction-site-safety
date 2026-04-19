#!/usr/bin/env python3
"""
Layer 2: GroundingDINO + SAM2 auto-annotation
20-class ontology: MOCS 13 + PPE 7
Output: annotations/pseudo_labels_coco.json
Resume-safe.
"""

import json, os, torch, numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# GroundingDINO
from groundingdino.util.inference import load_model, load_image, predict

# SAM2
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

BASE       = Path("/Data1/cse_24203016/construction_site/data/constructionsite10k")
GDINO_CFG  = "/Data1/cse_24203016/construction_site/models/groundingdino/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = "/Data1/cse_24203016/construction_site/models/groundingdino/groundingdino_swint_ogc.pth"
SAM2_CFG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT  = "/Data1/cse_24203016/construction_site/models/sam2/sam2.1_hiera_large.pt"
OUT_FILE   = BASE / "annotations/pseudo_labels_coco.json"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# 20-class ontology
CLASSES = [
    # MOCS 13
    "worker", "tower crane", "hanging hook", "truck crane", "roller",
    "bulldozer", "excavator", "truck", "loader", "pump truck",
    "concrete truck", "pile driver", "other vehicle",
    # PPE 7
    "hard hat", "safety vest", "safety harness", "gloves",
    "safety boots", "face mask", "safety goggles",
]
TEXT_PROMPT = " . ".join(CLASSES) + " ."

CONF_THRESH = 0.35
BOX_THRESH  = 0.35
MIN_AREA    = 400   # px²

print(f"Device: {DEVICE}")
print("Loading GroundingDINO ...")
gdino = load_model(GDINO_CFG, GDINO_CKPT, device=DEVICE)

print("Loading SAM2 ...")
sam2_model = build_sam2(SAM2_CFG, SAM2_CKPT, device=DEVICE)
sam2       = SAM2ImagePredictor(sam2_model)
print("Models loaded ✅")

# Load all image records
records = []
for split in ["train", "test"]:
    with open(BASE / f"annotations/{split}.json") as f:
        data = json.load(f)
    for r in data:
        r["_split"] = split
    records.extend(data)

# Resume
if OUT_FILE.exists():
    with open(OUT_FILE) as f:
        coco_out = json.load(f)
    done_ids = {img["file_name"] for img in coco_out["images"]}
    ann_id   = max((a["id"] for a in coco_out["annotations"]), default=0) + 1
else:
    coco_out = {"images": [], "annotations": [], "categories": [
        {"id": i+1, "name": c} for i, c in enumerate(CLASSES)
    ]}
    done_ids = set()
    ann_id   = 1

todo = [r for r in records if r["file_name"] not in done_ids]
print(f"To annotate: {len(todo)}")

SAVE_EVERY = 100

for idx, r in enumerate(tqdm(todo, desc="Auto-annotating")):
    img_path = str(BASE / "images" / r["_split"] / r["file_name"])
    try:
        image_src, image_transformed = load_image(img_path)
        H, W = image_src.shape[:2]
    except Exception as e:
        print(f"  ⚠️  {r['file_name']}: {e}")
        continue

    # GroundingDINO detection
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino,
            image=image_transformed,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESH,
            text_threshold=CONF_THRESH,
            device=DEVICE,
        )

    img_id = len(coco_out["images"]) + 1
    coco_out["images"].append({
        "id": img_id, "file_name": r["file_name"],
        "width": W, "height": H, "split": r["_split"]
    })

    if len(boxes) == 0:
        done_ids.add(r["file_name"])
        continue

    # Convert boxes to pixel coords (cx,cy,w,h → x1,y1,x2,y2)
    boxes_px = boxes.clone()
    boxes_px[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * W
    boxes_px[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * H
    boxes_px[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * W
    boxes_px[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * H
    boxes_px = boxes_px.cpu().numpy()

    # SAM2 segmentation
    pil_img = Image.open(img_path).convert("RGB")
    sam2.set_image(np.array(pil_img))

    for box, phrase, conf in zip(boxes_px, phrases, logits.tolist()):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw * bh < MIN_AREA:
            continue

        # Map phrase to class id
        phrase_lower = phrase.lower()
        cat_id = 13  # default: other vehicle
        for i, cls in enumerate(CLASSES):
            if cls in phrase_lower or phrase_lower in cls:
                cat_id = i + 1
                break

        # SAM2 mask
        try:
            masks, _, _ = sam2.predict(
                box=np.array([x1, y1, x2, y2]),
                multimask_output=False,
            )
            mask = masks[0].astype(np.uint8)
            # RLE-lite: store as bbox + area only (full RLE needs pycocotools)
            seg_area = int(mask.sum())
        except Exception:
            seg_area = int(bw * bh)

        coco_out["annotations"].append({
            "id":          ann_id,
            "image_id":    img_id,
            "category_id": cat_id,
            "bbox":        [float(x1), float(y1), float(bw), float(bh)],
            "area":        seg_area,
            "score":       float(conf),
            "phrase":      phrase,
            "iscrowd":     0,
        })
        ann_id += 1

    done_ids.add(r["file_name"])

    if (idx + 1) % SAVE_EVERY == 0:
        with open(OUT_FILE, "w") as f:
            json.dump(coco_out, f)

# Final save
with open(OUT_FILE, "w") as f:
    json.dump(coco_out, f)

total_ann = len(coco_out["annotations"])
print(f"\n✅ Done. {len(coco_out['images'])} images, {total_ann} annotations → {OUT_FILE}")
