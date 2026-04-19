#!/usr/bin/env python3
"""Generate captions with LoRA model then evaluate."""
import json, os, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rs
from tqdm import tqdm

BASE      = "/Data1/cse_24203016/construction_site"
BASE_DIR  = f"{BASE}/models/florence2-base"
LORA_DIR  = f"{BASE}/models/florence2-lora-finetuned"
CAP_JSON  = f"{BASE}/data/constructionsite10k/annotations/captions_florence2.json"
IMG_BASE  = Path(f"{BASE}/data/constructionsite10k/images")
OUT_DIR   = f"{BASE}/outputs"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
print("Loading LoRA model...")
processor = AutoProcessor.from_pretrained(LORA_DIR, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_DIR, torch_dtype=torch.float16, trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model = model.to(DEVICE).eval()
print("Model loaded ✅")

with open(CAP_JSON) as f:
    data = json.load(f)
test_data = [r for r in data if r.get("split") == "test"][:500]  # 500 samples for speed
print(f"Evaluating on {len(test_data)} test samples...")

gts, hyps = [], []
for r in tqdm(test_data, desc="Generating"):
    fname    = r["file_name"]
    split    = r.get("split", "test")
    img_path = IMG_BASE / split / fname
    if not img_path.exists():
        hits = list(IMG_BASE.rglob(fname))
        if not hits: continue
        img_path = hits[0]

    img  = Image.open(img_path).convert("RGB")
    enc  = processor(text="<DETAILED_CAPTION>", images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            pixel_values=enc["pixel_values"].to(torch.float16),
            max_new_tokens=128, do_sample=False,
        )
    caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
    gts.append(r["gt_caption"])
    hyps.append(caption)

print(f"\nGenerated {len(hyps)} captions")

# Metrics
sf = SmoothingFunction().method1
refs_tok = [[g.lower().split()] for g in gts]
hyps_tok = [h.lower().split() for h in hyps]
bleu4   = corpus_bleu(refs_tok, hyps_tok, weights=(0.25,)*4, smoothing_function=sf) * 100

meteor_scores = [meteor_score([g.lower().split()], h.lower().split())
                 for g, h in zip(gts, hyps)]
meteor  = sum(meteor_scores)/len(meteor_scores)*100

scorer  = rs.RougeScorer(["rougeL"], use_stemmer=True)
rl_scores = [scorer.score(g, h)["rougeL"].fmeasure for g, h in zip(gts, hyps)]
rouge_l = sum(rl_scores)/len(rl_scores)*100

scores = {"BLEU-4": round(bleu4,2), "METEOR": round(meteor,2), "ROUGE-L": round(rouge_l,2)}

print("\n" + "="*60)
print("LoRA Fine-tuned Florence-2 vs Base Model")
print("="*60)
print(f"{'Metric':<15} {'Base Model':>12} {'LoRA (ours)':>12}")
print("-"*60)
base = {"BLEU-4": 6.56, "METEOR": 21.94, "ROUGE-L": 24.54}
for k in ["BLEU-4", "METEOR", "ROUGE-L"]:
    delta = scores[k] - base[k]
    arrow = "↑" if delta > 0 else "↓"
    print(f"{k:<15} {base[k]:>12.2f} {scores[k]:>12.2f}  {arrow}{abs(delta):.2f}")
print("="*60)

# Save sample outputs
samples = [{"gt": gts[i], "lora": hyps[i]} for i in range(min(10, len(hyps)))]
result = {"lora_scores": scores, "base_scores": base,
          "n_evaluated": len(hyps), "sample_outputs": samples}
out_path = f"{OUT_DIR}/lora_caption_evaluation.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n✅ Saved to {out_path}")
