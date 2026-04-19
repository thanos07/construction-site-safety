from ultralytics import YOLO
import json

BASE   = "/Data1/cse_24203016/construction_site"
MODEL  = f"{BASE}/experiments/yolov8n_construction/weights/best.pt"
YAML   = f"{BASE}/datasets/construction.yaml"

model   = YOLO(MODEL)
metrics = model.val(data=YAML, split="val", device=0, batch=16)

# -- Your results -------------------------------------------------------------
your_map50    = float(metrics.box.map50)
your_map5095  = float(metrics.box.map)
your_precision = float(metrics.box.mp)
your_recall    = float(metrics.box.mr)

# -- MOCS paper baseline (Table 7 from the paper) -----------------------------
mocs_baselines = {
    "YOLO-v3 (Darknet53)":          {"mAP": 39.05, "AP50": 65.59, "FPS": 27.03},
    "Faster R-CNN (ResNet50+FPN)":  {"mAP": 50.64, "AP50": 74.65, "FPS":  8.39},
    "Mask R-CNN (ResNet50+FPN)":    {"mAP": 50.83, "AP50": 74.89, "FPS":  7.66},
    "PointRend (ResNet50+FPN)":     {"mAP": 51.04, "AP50": 74.79, "FPS":  8.87},
    "TridentFast (ResNet50)":       {"mAP": 50.69, "AP50": 73.08, "FPS":  4.05},
}

print("\n" + "="*65)
print("COMPARISON: Your YOLOv8n vs MOCS Paper Benchmarks")
print("="*65)
print(f"{'Model':<35} {'mAP':>6} {'AP50':>6} {'FPS':>6}")
print("-"*65)

# MOCS baselines
for name, vals in mocs_baselines.items():
    print(f"{name:<35} {vals['mAP']:>6.2f} {vals['AP50']:>6.2f} {vals['FPS']:>6.1f}")

print("-"*65)
print(f"{'YOLOv8n (Yours)':<35} {your_map5095*100:>6.2f} {your_map50*100:>6.2f} {'TBD':>6}")
print("="*65)

out = {
    "your_model": {
        "mAP50": your_map50, "mAP50-95": your_map5095,
        "precision": your_precision, "recall": your_recall
    },
    "mocs_baselines": mocs_baselines
}
with open(f"{BASE}/outputs/comparison_with_mocs.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved ? {BASE}/outputs/comparison_with_mocs.json")