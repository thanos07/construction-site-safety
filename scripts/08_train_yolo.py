# YOLOv8n training script
from ultralytics import YOLO

BASE = "/Data1/cse_24203016/construction_site"

model = YOLO("yolov8n.pt")

results = model.train(
    data=f"{BASE}/datasets/construction.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    project=f"{BASE}/experiments",
    name="yolov8n_construction",
    patience=15,
    amp=True,
    workers=2,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    save=True,
    exist_ok=True,
    verbose=True,
    flipud=0.0,
    fliplr=0.5,
    degrees=5.0,
    mosaic=1.0
)
print(f"Best model: {results.save_dir}/weights/best.pt")
