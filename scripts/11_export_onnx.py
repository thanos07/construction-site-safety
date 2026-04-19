from ultralytics import YOLO
import onnxruntime as ort
import numpy as np, json, time, os

BASE       = "/Data1/cse_24203016/construction_site"
MODEL_PATH = f"{BASE}/experiments/yolov8n_construction/weights/best.pt"
os.makedirs(f"{BASE}/outputs/onnx", exist_ok=True)

# Export
model = YOLO(MODEL_PATH)
model.export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17)

onnx_path = MODEL_PATH.replace(".pt", ".onnx")
print(f"ONNX exported: {onnx_path}")

# Benchmark with ORT TensorRT provider
providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
sess = ort.InferenceSession(onnx_path, providers=providers)
active = sess.get_providers()[0]
print(f"Active provider: {active}")

dummy     = np.random.randn(1, 3, 640, 640).astype(np.float32)
inp_name  = sess.get_inputs()[0].name

# Warmup
[sess.run(None, {inp_name: dummy}) for _ in range(10)]

# Benchmark 100 runs
N = 100
t0 = time.perf_counter()
for _ in range(N): sess.run(None, {inp_name: dummy})
elapsed = time.perf_counter() - t0

lat_ms = elapsed/N*1000
fps    = N/elapsed

print(f"\n=== ONNX Benchmark ===")
print(f"Provider:     {active}")
print(f"Latency/img:  {lat_ms:.2f} ms")
print(f"Throughput:   {fps:.1f} FPS")

result = {"provider": active, "latency_ms": lat_ms, "fps": fps, "n_runs": N}
with open(f"{BASE}/outputs/onnx/benchmark.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved ? {BASE}/outputs/onnx/benchmark.json")