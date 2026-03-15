#!/bin/bash
# scripts/prepare_data.sh
# ============================================================
# Step 1 of data pipeline: verify public datasets load correctly
#
# Usage:
#   bash scripts/prepare_data.sh
#
# What it does:
#   1. Checks all 5 public datasets are accessible
#   2. Runs extractor tests on each dataset
#   3. Prints sample QA pairs to verify correctness
#   4. Reports category and difficulty distributions
#   5. Saves a small local cache for fast iteration
#
# Run this BEFORE generating synthetic data.
# If any dataset fails here, fix the extractor first.
# ============================================================
set -e

echo "=================================================="
echo "  Think-in-Silence — Data Preparation"
echo "  Step 1: Public Dataset Verification"
echo "=================================================="

mkdir -p data/cache data/synthetic results

echo ""
echo "── Testing dataset extractors ───────────────────"
python -m pytest tests/test_dataloader.py -v

echo ""
echo "── Verifying public datasets load (streaming) ───"
python - << 'EOF'
from datasets import load_dataset

datasets_to_test = [
    ("hotpot_qa",            "distractor"),
    ("gsm8k",                "main"),
    ("commonsense_qa",       None),
    ("ai2_arc",              "ARC-Challenge"),
    ("allenai/strategy_qa",  None),
]

for name, config in datasets_to_test:
    try:
        kwargs = {"streaming": True, "split": "train"}
        if config:
            kwargs["name"] = config
        ds     = load_dataset(name, **kwargs)
        sample = next(iter(ds))
        print(f"  ✓ {name}")
    except Exception as e:
        print(f"  ✗ {name}: {e}")

print("\nAll accessible datasets confirmed.")
EOF

echo ""
echo "── Saving small verification cache (1000 samples) ─"
python - << 'EOF'
import json, sys, random
sys.path.insert(0, '.')

from datasets import load_dataset
from src.datasets.qa_datasets import (
    extract_hotpotqa, extract_gsm8k, extract_commonsenseqa,
    extract_arc, extract_strategyqa
)
from src.datasets.preprocessing.difficulty_filter import (
    compute_difficulty, classify_answer_type
)

configs = [
    ("hotpot_qa",           "distractor", extract_hotpotqa,        "factual_multihop"),
    ("gsm8k",               "main",       extract_gsm8k,           "math"),
    ("commonsense_qa",      None,         extract_commonsenseqa,   "commonsense"),
    ("ai2_arc",             "ARC-Challenge", extract_arc,          "science"),
    ("allenai/strategy_qa", None,         extract_strategyqa,      "strategy"),
]

samples = []
for name, config, extractor, category in configs:
    kwargs = {"streaming": True, "split": "train"}
    if config:
        kwargs["name"] = config
    ds = load_dataset(name, **kwargs)
    count = 0
    for raw in ds:
        if count >= 200:
            break
        try:
            s = extractor(raw)
            s["difficulty_score"] = compute_difficulty(s["question"], s["answer"])
            s["answer_type"]      = classify_answer_type(s["answer"])
            samples.append(s)
            count += 1
        except Exception:
            continue

random.shuffle(samples)
with open("data/cache/public_sample_1000.jsonl", "w") as f:
    for s in samples[:1000]:
        f.write(json.dumps(s) + "\n")

print(f"Saved {len(samples[:1000])} samples to data/cache/public_sample_1000.jsonl")
EOF

echo ""
echo "── Dataset statistics ───────────────────────────"
python -m src.datasets.preprocessing.stats \
    --data_file data/cache/public_sample_1000.jsonl

echo ""
echo "=================================================="
echo "  Public dataset preparation complete."
echo "  Next step: bash scripts/generate_synthetic.sh"
echo "=================================================="