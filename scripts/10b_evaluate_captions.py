#!/usr/bin/env python3
import json, os
from pathlib import Path
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rs
import nltk

BASE     = "/Data1/cse_24203016/construction_site"
CAP_JSON = f"{BASE}/data/constructionsite10k/annotations/captions_florence2.json"
OUT_DIR  = f"{BASE}/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading captions...")
with open(CAP_JSON) as f:
    data = json.load(f)

test_data = [r for r in data if r.get("split") == "test"]
print(f"Test samples: {len(test_data)}")

gts  = [r["gt_caption"]       for r in test_data]
hyps = [r["florence2_caption"] for r in test_data]

# ── BLEU-4 ───────────────────────────────────────────────────────────────────
print("Computing BLEU-4...")
refs_tok = [[gt.lower().split()] for gt in gts]
hyps_tok = [h.lower().split() for h in hyps]
sf = SmoothingFunction().method1
bleu4 = corpus_bleu(refs_tok, hyps_tok,
                    weights=(0.25,0.25,0.25,0.25),
                    smoothing_function=sf) * 100
print(f"  BLEU-4: {bleu4:.2f}")

# ── METEOR ───────────────────────────────────────────────────────────────────
print("Computing METEOR...")
meteor_scores = []
for gt, hyp in zip(gts, hyps):
    try:
        score = meteor_score([gt.lower().split()], hyp.lower().split())
        meteor_scores.append(score)
    except:
        pass
meteor = (sum(meteor_scores)/len(meteor_scores)*100) if meteor_scores else 0
print(f"  METEOR: {meteor:.2f}")

# ── ROUGE-L ──────────────────────────────────────────────────────────────────
print("Computing ROUGE-L...")
scorer = rs.RougeScorer(["rougeL"], use_stemmer=True)
rl_scores = [scorer.score(gt, hyp)["rougeL"].fmeasure
             for gt, hyp in zip(gts, hyps)]
rouge_l = sum(rl_scores)/len(rl_scores)*100
print(f"  ROUGE-L: {rouge_l:.2f}")

# ── BERTScore ────────────────────────────────────────────────────────────────
print("Computing BERTScore...")
try:
    from bert_score import score as bscore
    P, R, F1 = bscore(hyps, gts, lang="en", verbose=False)
    bertscore = F1.mean().item() * 100
    print(f"  BERTScore-F1: {bertscore:.2f}")
except Exception as e:
    bertscore = 0
    print(f"  BERTScore error: {e}")

scores = {
    "BLEU-4":       round(bleu4, 2),
    "METEOR":       round(meteor, 2),
    "ROUGE-L":      round(rouge_l, 2),
    "BERTScore-F1": round(bertscore, 2),
}

# ── Comparison table ─────────────────────────────────────────────────────────
# NOTE: Chen & Zou report BERTScore on a different scale (their ~37 = our ~88 equivalent)
# We report our BERTScore in standard library scale for reproducibility
print("\n" + "="*72)
print("CAPTION EVALUATION — Florence-2 base vs Chen & Zou (2025) Table 5")
print("="*72)
print(f"{'Model':<30} {'BLEU-4':>7} {'METEOR':>7} {'ROUGE-L':>8}")
print("-"*72)
baselines = [
    ("GPT-4V zero-shot",  None, 33.7, None),
    ("GPT-4o 5-shot",     None, 39.4, None),
    ("LLaVA-13B",         None, 29.8, None),
]
for name, b4, met, rl in baselines:
    b4s  = f"{b4:.1f}"  if b4  else "  —"
    mets = f"{met:.1f}" if met else "  —"
    rls  = f"{rl:.1f}"  if rl  else "  —"
    print(f"{name:<30} {b4s:>7} {mets:>7} {rls:>8}")
print("-"*72)
print(f"{'Florence-2 base (ours)':<30} "
      f"{scores['BLEU-4']:>7.2f} "
      f"{scores['METEOR']:>7.2f} "
      f"{scores['ROUGE-L']:>8.2f}")
print("="*72)
print(f"\nBERTScore-F1 (standard scale 0-100): {scores['BERTScore-F1']:.2f}")
print("Note: Chen & Zou BERTScore uses different normalization — not directly comparable")

result = {"our_scores": scores, "n_test": len(test_data),
          "note": "BERTScore in standard bert-score library scale (0-100)"}
with open(f"{OUT_DIR}/caption_evaluation.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"\n✅ Saved to {OUT_DIR}/caption_evaluation.json")
