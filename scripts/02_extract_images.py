#!/usr/bin/env python3
import io, json
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pathlib import Path
from tqdm import tqdm

BASE    = Path("/Data1/cse_24203016/construction_site/data/constructionsite10k")
ANN_DIR = BASE / "annotations"
ANN_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": [
        BASE / "train-00001-of-00002.parquet",
        BASE / "train-00002-of-00002.parquet",
    ],
    "test": [BASE / "test.parquet"],
}

class NumpyEncoder(json.JSONEncoder):
    """Handles numpy scalars, arrays, and pandas NA inside json.dump."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        try:
            import pandas as pd
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return super().default(obj)

def safe_rule(row, key):
    val = row[key]
    if val is None:
        return {"bboxes": [], "reason": None}
    return {
        "bboxes": val.get("bounding_box"),   # encoder handles numpy
        "reason": val.get("reason"),
    }

def extract_split(split_name, parquet_files):
    out_img_dir = BASE / "images" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0

    for parquet_file in parquet_files:
        print(f"\n📦 Reading {parquet_file.name} ...")
        table = pq.read_table(parquet_file)
        df    = table.to_pandas()
        print(f"   Rows: {len(df)}")

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            img_struct = row["image"]
            if img_struct is None:
                skipped += 1
                continue

            raw_bytes = img_struct.get("bytes")
            fname     = Path(img_struct.get("path", f"{idx:07d}.jpg")).name
            out_path  = out_img_dir / fname

            if not out_path.exists():
                try:
                    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    img.save(out_path, "JPEG", quality=95)
                except Exception as e:
                    print(f"  ⚠️  {fname}: {e}")
                    skipped += 1
                    continue

            records.append({
                "file_name":       fname,
                "image_id":        row.get("image_id"),
                "caption":         row.get("image_caption"),
                "illumination":    row.get("illumination"),
                "camera_distance": row.get("camera_distance"),
                "view":            row.get("view"),
                "quality_of_info": row.get("quality_of_info"),
                "rule_1":          safe_rule(row, "rule_1_violation"),
                "rule_2":          safe_rule(row, "rule_2_violation"),
                "rule_3":          safe_rule(row, "rule_3_violation"),
                "rule_4":          safe_rule(row, "rule_4_violation"),
                "excavator":                  row.get("excavator"),
                "rebar":                      row.get("rebar"),
                "worker_with_white_hard_hat": row.get("worker_with_white_hard_hat"),
            })

    ann_path = ANN_DIR / f"{split_name}.json"
    with open(ann_path, "w") as f:
        json.dump(records, f, cls=NumpyEncoder)

    img_count = len(list(out_img_dir.glob("*.jpg")))
    print(f"\n✅ {split_name}: {img_count} images  →  {out_img_dir}")
    print(f"✅ {split_name}: {len(records)} records  →  {ann_path}")
    if skipped:
        print(f"⚠️  {skipped} rows skipped")

for split, files in SPLITS.items():
    extract_split(split, files)

print("\n🎉 All done.")
