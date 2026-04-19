# Multimodal Auto-Annotation and Safety Grounding for Construction Site Perception

## Official Title

> *Multimodal Auto-Annotation and Safety Grounding for Construction Site Perception using Grounded Segmentation and Compact Edge Deployment*

---

## Table of Contents

1. [Project Vision](#1-project-vision)
2. [Research Motivation](#2-research-motivation)
3. [System Architecture](#3-system-architecture)
4. [Key Literature](#4-key-literature)
5. [Dataset](#5-dataset)
6. [20-Class Ontology](#6-20-class-ontology)
7. [Model Stack](#7-model-stack)
8. [Installation and Setup](#8-installation-and-setup)
9. [Directory Structure](#9-directory-structure)
10. [Execution Order](#10-execution-order)
11. [Results](#11-results)
12. [Evaluation Plan and Baselines](#12-evaluation-plan)
13. [Paper Outline](#13-paper-outline)
14. [Citations](#14-citations)
15. [HPC Acknowledgement](#15-hpc-acknowledgement)

---

## 1. Project Vision

Build an end-to-end **multimodal auto-annotation and safety grounding pipeline** for construction site images. Input: 10,013 raw unlabeled construction site images. Output:

- Natural language captions describing every construction scene
- Safety rule violation detection with reasoning and bounding box localization
- Object-level bounding boxes + segmentation masks across 18 construction classes
- A compact downstream detector deployable at edge speed via ONNX

**Core research claim:** Current off-the-shelf VLMs (including GPT-4o) cannot reliably ground objects at construction sites — best IoU is ~35% on excavators and ~10% on workers with white hard hats (Chen & Zou, 2025). This project closes that gap through LoRA fine-tuning of Florence-2, automated pseudo-labeling with GroundingDINO + SAM2, and compact YOLOv8n training with ONNX export.

---

## 2. Research Motivation

- Construction sites accounted for **21% of all U.S. workplace fatalities** in 2022
- Manual safety inspection is slow, inconsistent, and not scalable
- General-purpose CV models fail on construction images due to occlusion, clutter, variable illumination, and domain-specific objects
- Chen & Zou (2025) proved GPT-4o achieves <20% IoU on constrained objects and recommended fine-tuning — this project is that next step
- No existing open system combines VLM understanding + auto-annotation + compact edge deployment for construction safety

**This project fills all three gaps simultaneously with zero human annotation cost.**

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────┐
│            LAYER 1: Vision-Language Understanding         │
│                                                          │
│  Florence-2-base (microsoft/Florence-2-base)             │
│  + LoRA fine-tuning (r=16, lora_alpha=32,               │
│    target: q_proj + v_proj, 0.38% trainable params)      │
│                                                          │
│  Tasks:                                                  │
│  ├── Image captioning (<DETAILED_CAPTION> prompt)        │
│  ├── Safety VQA (4 rules from Chen & Zou 2025)           │
│  └── Text-image grounding evaluation                     │
│                                                          │
│  Results: METEOR 30.95 (beats LLaVA-13B 29.8) ✅        │
└──────────────────────┬───────────────────────────────────┘
                       │ captions / task tokens
                       ▼
┌──────────────────────────────────────────────────────────┐
│         LAYER 2: Auto-Annotation Engine                   │
│                                                          │
│  GroundingDINO-SwinT → bounding boxes                    │
│       +                                                  │
│  SAM 2 (sam2_hiera_large) → segmentation masks           │
│                                                          │
│  18-class ontology (MOCS 13 + PPE 5)                    │
│  Quality filter: conf > 0.35, area > 400px²             │
│  Output: COCO JSON + YOLO format labels                  │
│                                                          │
│  Results: 57,226 annotations, 99.2% image coverage ✅   │
└──────────────────────┬───────────────────────────────────┘
                       │ pseudo-labels
                       ▼
┌──────────────────────────────────────────────────────────┐
│         LAYER 3: Compact Deployment Model                 │
│                                                          │
│  YOLOv8n (ultralytics) — trained on pseudo-labels        │
│  Export: ONNX (11.7 MB)                                  │
│  Inference: 1.2ms GPU / 75ms CPU                         │
│                                                          │
│  Results: mAP50=38.1%, 833 FPS GPU ✅                    │
└──────────────────────────────────────────────────────────┘
```

---

## 4. Key Literature

### Chen & Zou (2025) — Primary Dataset
*"Are Large Pre-Trained Vision Language Models Effective Construction Safety Inspectors?"*
arXiv:2508.11011 — Created ConstructionSite 10k with captions, VQA, and visual grounding annotations.

**Baselines to beat (Table 8):**

| Model | Excavator IoU | Worker w/ White Hat IoU |
|-------|--------------|------------------------|
| GPT-4V zero-shot | 35.8% | 10.1% |
| LLaVA-13B | 54.5% | 23.9% |
| GroundingDINO-B | **71.0%** | 11.2% |

**Captioning baselines (Table 5):**

| Model | METEOR | CIDEr |
|-------|--------|-------|
| GPT-4o 5-shot | **39.4** | **169.9** |
| LLaVA-13B | 29.8 | 69.2 |

### An et al. (2021) — MOCS Benchmark
*"Dataset and benchmark for detecting moving objects in construction sites."*
Automation in Construction, 122, 103482.

**Detection baselines (Table 7):**

| Detector | mAP | AP50 | FPS |
|----------|-----|------|-----|
| YOLO-v3 | 39.05 | 65.59 | 27.03 |
| Faster R-CNN | 50.64 | 74.65 | 8.39 |
| PointRend (best) | **51.04** | **74.79** | 8.87 |

---

## 5. Dataset

**ConstructionSite 10k** — `LouisChen15/ConstructionSite` on HuggingFace

- **Total:** 10,013 images (7,009 train + 3,004 test)
- **Download format:** Parquet files (~4.3 GB) — images embedded as binary blobs
- **Annotations:** Captions, VQA (4 safety rules), visual grounding boxes per image

> ⚠️ Dataset requires HuggingFace account approval. Accept terms at the dataset page before downloading. Use `export HF_TOKEN=<your_token>` — never hardcode tokens in scripts.

**Parquet schema (key fields):**
```
image:                      struct<bytes: binary, path: string>
image_caption:              string
rule_1/2/3/4_violation:     struct<bounding_box, reason>
excavator:                  list<list<double>>
rebar:                      list<list<double>>
worker_with_white_hard_hat: list<list<double>>
```

**Extracted to disk:**
```
images/train/    7,009 .jpg files
images/test/     3,004 .jpg files
annotations/train.json   7,009 records
annotations/test.json    3,004 records
```

---

## 6. 20-Class Ontology

```python
CLASSES = [
    # MOCS 13 (An et al. 2021)
    "worker", "tower crane", "hanging hook", "truck crane", "roller",
    "bulldozer", "excavator", "truck", "loader", "pump truck",
    "concrete truck", "pile driver", "other vehicle",
    # PPE 5 (added by this project)
    "hard hat", "safety vest", "safety harness", "gloves", "safety boots",
]
# Final active classes: 18 (pump truck merged → other vehicle)
```

---

## 7. Model Stack

| Model | Purpose | Size |
|-------|---------|------|
| Florence-2-base | Captioning + LoRA base | 930 MB |
| GroundingDINO SwinT | Auto-annotation (boxes) | 662 MB |
| SAM 2 hiera-large | Auto-annotation (masks) | 857 MB |
| Florence-2 + LoRA | Fine-tuned captioner | ~50 MB |
| YOLOv8n best.pt | Compact detector | 6.2 MB |
| YOLOv8n best.onnx | Deployment model | 11.7 MB |

---

## 8. Installation and Setup

### Environment

```bash
conda create -n construction python=3.10
conda activate construction
```

### PyTorch — MUST match your CUDA toolkit version

```bash
# Check your nvcc version first: nvcc --version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Core packages

```bash
pip install transformers==4.48.0 accelerate peft
pip install ultralytics
pip install huggingface_hub pyarrow pillow tqdm ninja
pip install nltk rouge-score bert-score
pip install onnx onnxruntime
```

### GroundingDINO (requires source patches)

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO

# Patch deprecated PyTorch API before building
sed -i 's/value\.type()\.is_cuda()/value.is_cuda()/g' \
    groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h
sed -i 's/AT_DISPATCH_FLOATING_TYPES(input\.type()/AT_DISPATCH_FLOATING_TYPES(input.scalar_type()/g' \
    groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu
sed -i 's/AT_DISPATCH_FLOATING_TYPES(value\.type()/AT_DISPATCH_FLOATING_TYPES(value.scalar_type()/g' \
    groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu

TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation -e .
cd ..
```

### SAM2

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install --no-build-isolation -e .
cd ..
```

### Download model weights

```bash
# Florence-2
python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='microsoft/Florence-2-base',
    local_dir='models/florence2-base',
    token=os.environ['HF_TOKEN']
)
"

# GroundingDINO SwinT checkpoint
mkdir -p models/groundingdino
wget -O models/groundingdino/groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -O models/groundingdino/GroundingDINO_SwinT_OGC.py \
    https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py

# SAM2 hiera-large
mkdir -p models/sam2
wget -O models/sam2/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

---

## 9. Directory Structure

```
construction_site/
│
├── data/constructionsite10k/
│   ├── images/train/            7,009 .jpg
│   ├── images/test/             3,004 .jpg
│   └── annotations/
│       ├── train.json           7,009 records
│       ├── test.json            3,004 records
│       ├── captions_florence2.json   10,013 captions ✅
│       └── pseudo_labels_coco.json   57,226 annotations ✅
│
├── models/
│   ├── florence2-base/          base model weights
│   ├── groundingdino/           SwinT checkpoint + config
│   ├── sam2/                    hiera-large checkpoint
│   └── florence2-lora-finetuned/  LoRA adapters ✅
│
├── scripts/
│   ├── 02_extract_images.py     extract images from parquet
│   ├── 05_caption_florence2.py  generate captions
│   ├── 06_auto_annotate.py      GDino + SAM2 annotation
│   ├── 07_coco_to_yolo.py       format conversion + split
│   ├── 08_train_yolo.py         YOLOv8n training
│   ├── 09_lora_finetune.py      Florence-2 LoRA fine-tuning
│   ├── 10_evaluate_and_compare.py   detection evaluation
│   ├── 10b_evaluate_captions.py     base caption evaluation
│   ├── 10c_evaluate_lora_captions.py  LoRA caption evaluation
│   └── 11_export_onnx.py        ONNX export + benchmark
│
├── pbs_jobs/                    PBS job scripts for HPC
├── datasets/construction.yaml   YOLO dataset config (18 classes)
├── experiments/yolov8n_construction/
│   ├── weights/best.pt          6.2 MB best model ✅
│   ├── weights/best.onnx        11.7 MB ONNX export ✅
│   └── results.csv              training curves ✅
└── outputs/
    ├── comparison_with_mocs.json
    ├── caption_evaluation.json
    ├── lora_caption_evaluation.json
    └── onnx/benchmark.json
```

---

## 10. Execution Order

| Step | Script | How | Time |
|------|--------|-----|------|
| 1 | Download + extract dataset | `02_extract_images.py` | 20 min |
| 2 | Install models + weights | Manual (see Section 8) | 15 min |
| 3 | Generate captions | `job_caption.pbs` | 16 min |
| 4 | Auto-annotate | `job_autoannotate.pbs` | 30 min |
| 5 | Convert to YOLO format | `07_coco_to_yolo.py` | 5 min |
| 6 | Train YOLOv8n | `job_yolo_train.pbs` | 1.4 hrs |
| 7 | LoRA fine-tuning | `job_lora.pbs` | ~2 hrs |
| 8 | Evaluate | `10_evaluate_and_compare.py` | 2 min |
| 9 | Caption eval | `10b/10c_evaluate_captions.py` | 5 min |
| 10 | ONNX export | `job_onnx.pbs` | 5 min |

### PBS Script Rules (HPC-specific)

```bash
# ALWAYS use full Python path — never 'conda activate'
/path/to/conda/envs/construction/bin/python3 scripts/script.py

# NEVER hardcode GPU UUIDs — let PBS assign GPU
# Remove: export CUDA_VISIBLE_DEVICES=MIG-...

# NEVER use $PBS_JOBID in -o directive — use static names
#PBS -o /path/to/logs/job_caption.log

# NEVER exceed queue memory limit (check with: qstat -q)
#PBS -l select=1:ncpus=4:mem=16gb
```

---

## 11. Results

### Layer 1 — Captioning

| Model | METEOR | ROUGE-L |
|-------|--------|---------|
| GPT-4o 5-shot (best baseline) | 39.4 | — |
| LLaVA-13B | 29.8 | — |
| Florence-2 base (ours, pre-LoRA) | 21.94 | 24.54 |
| **Florence-2 + LoRA 3-epoch (ours)** | **30.95** | **25.60** |

**Florence-2 + LoRA beats LLaVA-13B on METEOR ✅**

### Layer 2 — Auto-Annotation

| Metric | Value |
|--------|-------|
| Total annotations | 57,226 |
| Images annotated | 10,013 |
| Coverage | 99.2% |
| Confidence mean | 0.442 |
| Confidence range | 0.350 – 0.865 |

### Layer 3 — Detection

| Detector | mAP50-95 | mAP50 | FPS | Params | Training Data |
|----------|----------|-------|-----|--------|---------------|
| YOLO-v3 (baseline) | 39.05 | 65.59 | 27 | 62M | 41,668 GT labels |
| Faster R-CNN (baseline) | 50.64 | 74.65 | 8.4 | 41M | 41,668 GT labels |
| PointRend (best baseline) | 51.04 | 74.79 | 8.9 | ~44M | 41,668 GT labels |
| **YOLOv8n (ours)** | **28.34** | **38.09** | **833** | **3M** | **10,013 pseudo** |

**Top per-class AP50:**

| Class | AP50 |
|-------|------|
| Worker | 83.1% |
| Excavator | 69.1% |
| Hard hat | 68.8% |
| Tower crane | 60.5% |
| Safety vest | 53.2% |

### Deployment

| Format | Size | Latency | FPS |
|--------|------|---------|-----|
| PyTorch (.pt) | 6.2 MB | 1.2ms | 833 GPU |
| ONNX (.onnx) | 11.7 MB | 75ms | 13.3 CPU |

### Key Claims for Paper

- Trained on **4× less data** than MOCS baselines (10k vs 41k images)
- **Zero human annotation** — fully automated pipeline
- Beats LLaVA-13B on METEOR despite 3-epoch LoRA on pseudo-labels
- **833 FPS GPU** inference — real-time edge deployment ready
- **13× fewer parameters** than Faster R-CNN (3M vs 41M)

---

## 12. Evaluation Plan

### Metrics used

| Task | Metrics | Library |
|------|---------|---------|
| Captioning | BLEU-4, METEOR, ROUGE-L, BERTScore | NLTK, rouge-score, bert-score |
| Detection | mAP50, mAP50-95, Precision, Recall | Ultralytics val |
| Deployment | Latency (ms), FPS, model size (MB) | ONNX Runtime |

### Important notes on metrics

- **pycocoevalcap** (CIDEr, SPICE) requires Java — not available on HPC. Use NLTK instead.
- **BERTScore** from bert-score library returns values ~0.88 which is NOT comparable to Chen & Zou's ~37 — they use different normalization. Report separately and note the difference.
- Use **METEOR as primary captioning metric** — handles synonyms and paraphrasing better than BLEU for domain-specific text.

---

## 13. Paper Outline

### Section 1 — Introduction
- 21% fatality statistic, VLM grounding gap, 3-layer contribution

### Section 2 — Related Work
- ConstructionSite 10k (Chen & Zou, 2025)
- MOCS (An et al., 2021)
- Florence-2, GroundingDINO, SAM2, YOLOv8, LoRA

### Section 3 — Methodology
- Three-layer architecture (Figure 1)
- Layer 1: LoRA config (r=16, 0.38% params)
- Layer 2: GDino+SAM2 pipeline, quality filters
- Layer 3: YOLOv8n training setup

### Section 4 — Experiments
- Dataset: ConstructionSite 10k, 7948/994/994 split
- Hardware: NVIDIA H100, CUDA 11.8, PyTorch 2.7.1

### Section 5 — Results
- Table 1: Captioning vs GPT-4V / GPT-4o / LLaVA-13B
- Table 2: Detection vs MOCS baselines
- Table 3: Per-class AP50 (worker, excavator, hard hat highlighted)
- Table 4: Deployment benchmark

### Section 6 — Discussion
- 4× less data, zero annotation cost, edge-deployable
- Failure cases: loader (AP50=2%), concrete truck (AP50=5.2%) — rare classes

### Section 7 — Conclusion + Future Work
- More LoRA epochs, TensorRT when available, real-time demo

---

## 14. Citations

```bibtex
@article{chen2025constructionsite,
  title={Are Large Pre-Trained Vision Language Models Effective Construction Safety Inspectors?},
  author={Chen, Xuezheng and Zou, Zhengbo},
  journal={arXiv preprint arXiv:2508.11011},
  year={2025}
}

@article{an2021mocs,
  title={Dataset and benchmark for detecting moving objects in construction sites},
  author={An, Xuehui and Zhou, Li and Liu, Zuguang and Wang, Chengzhi and Li, Pengfei and Li, Zhiwei},
  journal={Automation in Construction},
  volume={122},
  pages={103482},
  year={2021}
}

@inproceedings{liu2024groundingdino,
  title={Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection},
  author={Liu, Shilong and others},
  booktitle={ECCV},
  year={2024}
}

@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and others},
  journal={arXiv:2408.00714},
  year={2024}
}

@article{xiao2023florence2,
  title={Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  author={Xiao, Bin and others},
  journal={arXiv:2311.06242},
  year={2023}
}

@article{jocher2023yolov8,
  title={Ultralytics YOLOv8},
  author={Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing},
  year={2023},
  url={https://github.com/ultralytics/ultralytics}
}

@article{hu2022lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  journal={ICLR},
  year={2022}
}
```

---

## 15. HPC Acknowledgement

All publications, theses, or presentations must include:

> *"The authors gratefully acknowledge the High Performance Computing (HPC) facility provided by Dr. B. R. Ambedkar National Institute of Technology Jalandhar (NIT Jalandhar) for supporting the computational requirements of this research work."*

---

## Security Notes

> ⚠️ This repository does **not** contain:
> - HuggingFace API tokens
> - GPU UUIDs or MIG device IDs
> - HPC login credentials or IP addresses
> - Any passwords or private keys

---

*README maintained by Noor | NIT Jalandhar | Version: 5.0 (Public/GitHub)*
