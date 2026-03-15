#!/bin/bash
# scripts/generate_synthetic.sh
# ============================================================
# Step 2 of data pipeline: generate synthetic multi-hop QA pairs
#
# Usage:
#   # Test run first (100 paragraphs, ~5 minutes)
#   bash scripts/generate_synthetic.sh --test
#
#   # Full run with local Mistral (2-3 days for 5M samples)
#   bash scripts/generate_synthetic.sh /path/to/corpus 5000000
#
#   # Smaller run (500K samples, ~12 hours on L4)
#   bash scripts/generate_synthetic.sh /path/to/corpus 500000
#
# Arguments:
#   $1 = corpus directory (default: /corpus)
#   $2 = target samples   (default: 500000)
#   --test = test run with 100 paragraphs only
#
# IMPORTANT:
#   Run this in tmux — it takes hours/days.
#   tmux new -s synthesis
#   bash scripts/generate_synthetic.sh /path/to/corpus 500000
#   Ctrl+B, D to detach
# ============================================================
set -e

CORPUS_DIR=${1:-data/corpus}
TARGET=${2:-500000}
TEST_FLAG=""

# Check for --test flag
for arg in "$@"; do
    if [ "$arg" = "--test" ]; then
        TEST_FLAG="--test_run"
        TARGET=1000
        echo "TEST RUN MODE: 100 paragraphs only"
    fi
done

echo "=================================================="
echo "  Synthetic Data Generation"
echo "  Corpus:  $CORPUS_DIR"
echo "  Target:  $TARGET samples"
echo "  Generator: local (Mistral-7B)"
echo "=================================================="

# Verify corpus directory exists
if [ ! -d "$CORPUS_DIR" ]; then
    echo "ERROR: Corpus directory not found: $CORPUS_DIR"
    echo ""
    echo "Your corpus should be a directory of text files."
    echo "Supported formats: .txt, .md, .jsonl (with 'text' field)"
    echo ""
    echo "To get started with a smaller corpus:"
    echo "  mkdir -p data/corpus"
    echo "  # Add your text files to data/corpus/"
    exit 1
fi

# Count corpus files
FILE_COUNT=$(find "$CORPUS_DIR" -type f \( -name "*.txt" -o -name "*.md" -o -name "*.jsonl" \) | wc -l)
echo "Found $FILE_COUNT corpus files in $CORPUS_DIR"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No .txt, .md, or .jsonl files found in $CORPUS_DIR"
    exit 1
fi

echo ""
echo "── Starting generation ──────────────────────────"
echo "Progress saved to: data/synthetic/progress.json"
echo "Output file: data/synthetic/synthetic_qa.jsonl"
echo "Safe to interrupt and resume anytime."
echo ""

python -m src.datasets.synthetic.pipeline \
    --corpus_dir "$CORPUS_DIR" \
    --output_dir data/synthetic \
    --generator local \
    --target_samples "$TARGET" \
    --min_difficulty 1 \
    $TEST_FLAG \
    2>&1 | tee data/synthetic/generation_log.txt

echo ""
echo "── Post-generation statistics ───────────────────"
if [ -f "data/synthetic/synthetic_qa.jsonl" ]; then
    python -m src.datasets.preprocessing.stats \
        --data_file data/synthetic/synthetic_qa.jsonl
fi

echo ""
echo "=================================================="
echo "  Generation complete."
echo "  Review the statistics above before training."
echo "  If multi-hop ratio < 0.70: run with stricter filtering."
echo "  Next step: open notebooks/02_data_exploration.ipynb"
echo "=================================================="