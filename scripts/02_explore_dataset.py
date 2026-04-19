# scripts/02_explore_dataset.py
import os, json
from pathlib import Path

BASE = "/Data1/cse_24203016/construction_site/data/constructionsite10k"

print("=== Files downloaded ===")
for p in Path(BASE).rglob("*"):
    if p.is_file() and p.suffix in ['.json', '.txt', '.csv', '.yaml']:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.relative_to(BASE)}  ({size_mb:.1f} MB)")

print("\n=== Image count ===")
imgs = list(Path(BASE).rglob("*.jpg")) + list(Path(BASE).rglob("*.png"))
print(f"  Total images: {len(imgs)}")

print("\n=== JSON files ===")
for jf in Path(BASE).rglob("*.json"):
    with open(jf) as f:
        data = json.load(f)
    if isinstance(data, dict):
        print(f"  {jf.name}: keys = {list(data.keys())[:5]}")
    elif isinstance(data, list):
        print(f"  {jf.name}: list of {len(data)} items, sample keys = {list(data[0].keys()) if data else '[]'}")